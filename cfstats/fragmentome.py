import pandas as pd 
import base64
import copy
import io
import os
import pickle
import tempfile
import uuid

import dash
from dash import dcc, html
from dash import dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import logging
import time
import warnings

from cfstats import fszd, csm, fpends, db

warnings.filterwarnings(
    'ignore',
    message=r'.*The copy keyword is deprecated and will be removed in a future version\..*',
    category=FutureWarning,
)

# Example dataframe
# df = pd.read_csv(...)
# Must contain columns: 'x', 'y'

def _prepare_meta(dfmeta):
    """Prepare meta DataFrame for the UI: categorize low-cardinality columns, extract color/numeric columns."""
    dfmeta['_sample_id'] = dfmeta.index
    numeric_columns = [c for c in dfmeta.columns if pd.api.types.is_numeric_dtype(dfmeta[c])]
    color_columns = []
    # Convert float64 columns to int when possible and distinct values ≤ 25
    for col in dfmeta.select_dtypes(include=['float64', 'int64', 'object']).columns:
        if col in ['x', 'y']:
            continue
        distinct_vals = dfmeta[col].dropna().unique()
        if len(distinct_vals) <= 25:
            dfmeta[col] = dfmeta[col].astype(str)  # nullable integer type
        color_columns.append(col)
    return dfmeta, numeric_columns, color_columns


def _prepare_core(dfcore):
    """Extract core, csm, and 5p column groups from the core DataFrame."""
    core_cols = []
    for c in dfcore.columns:
        try:
            ci = int(c)
        except (TypeError, ValueError):
            continue
        if 1 <= ci <= 1000:
            core_cols.append((ci, c))
    core_cols.sort(key=lambda t: t[0])
    core_x = [t[0] for t in core_cols]
    core_col_names = [t[1] for t in core_cols]
    csm_col_names = [c for c in dfcore.columns if str(c).endswith('-csm')]
    csm_x = [str(c)[:-4] for c in csm_col_names]
    p5_col_names = [c for c in dfcore.columns if str(c).endswith('-5p')]
    p5_x = [str(c)[:-3] for c in p5_col_names]
    return core_x, core_col_names, csm_col_names, csm_x, p5_col_names, p5_x


def explore(args):

    log = logging.getLogger(__name__)

    t0 = time.time()
    log.info('Connecting to ClickHouse')

    ch_client = db.get_client(
        host=getattr(args, 'ch_host', 'localhost'),
        port=getattr(args, 'ch_port', 8123)
    )

    # Check if we have data without loading full tables
    has_core_data = db.get_sample_count(ch_client, db.CORE_TABLE) > 0
    has_meta_data = db.get_sample_count(ch_client, db.META_TABLE) > 0
    has_data = has_core_data and has_meta_data
    
    log.info('Data check complete in %.2fs (has_core=%s, has_meta=%s)', time.time() - t0, has_core_data, has_meta_data)

    # Get feature names lazily for dropdowns
    meta_features = db.get_meta_features(ch_client) if has_meta_data else []
    numeric_columns = []  # Will be determined on-demand
    color_columns = meta_features  # All meta features available for coloring
    
    # Get core column groups lazily
    core_x, core_col_names, csm_col_names, csm_x, p5_col_names, p5_x = [], [], [], [], [], []
    if has_core_data:
        # Get core column names from schema
        fresh_client = db.get_fresh_client(
            host=getattr(args, 'ch_host', 'localhost'),
            port=getattr(args, 'ch_port', 8123)
        )
        columns_result = fresh_client.query(
            f"SELECT name FROM system.columns WHERE database='{db.DATABASE_NAME}' AND table='{db.CORE_TABLE}' AND name != 'sample_id' ORDER BY position"
        )
        if columns_result.result_rows:
            core_columns = [row[0] for row in columns_result.result_rows]
            # Extract core, csm, and 5p column groups
            for c in core_columns:
                try:
                    ci = int(c)
                except (TypeError, ValueError):
                    continue
                if 1 <= ci <= 1000:
                    core_x.append(ci)
                    core_col_names.append(c)
            
            csm_col_names = [c for c in core_columns if str(c).endswith('-csm')]
            csm_x = [str(c)[:-4] for c in csm_col_names]
            p5_col_names = [c for c in core_columns if str(c).endswith('-5p')]
            p5_x = [str(c)[:-3] for c in p5_col_names]
    
    # Determine numeric columns from meta features
    numeric_columns = meta_features  # Simplified - will be refined per-query

    default_x_col = None
    default_y_col = None

    mapping = None
    if getattr(args, 'mapping', None):
        mapping = pickle.load(open(args.mapping, 'rb'))
        reducer = mapping[0]
        embedding = mapping[1]
        mapping_k = mapping[2]
    
    admin_password = getattr(args, 'admin_password', None) or os.environ.get('FRAGMENTOME_ADMIN_PASSWORD')

    app = dash.Dash("Fragmentome explorer", suppress_callback_exceptions=True)

    upload_max_size = 1024 * 1024 * 1024
    app.server.config['MAX_CONTENT_LENGTH'] = upload_max_size

    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='selected-sample', data=None),
        dcc.Store(id='uploaded-point', data=None),
        dcc.Store(id='admin-auth', data=False),
        dcc.Store(id='core-tsv-bytes', data=None),
        dcc.Store(id='core-column-types', data=None),
        dcc.Store(id='filter-indices', data=[]),
        dcc.Store(id='filters-store', data=[]),
        dcc.Store(id='cloud-ids-core', data=[]),
        html.Div(id='page-content'),
    ])

    def _apply_filters(filters, client=None):
        """Apply filters using ClickHouse and return sample_ids.

        IMPORTANT: Dash callbacks can run concurrently; do not share a single
        clickhouse_connect client across callbacks.
        """
        if client is None:
            client = db.get_fresh_client(
                host=getattr(args, 'ch_host', 'localhost'),
                port=getattr(args, 'ch_port', 8123)
            )
        return db.apply_filters_clickhouse(client, filters)

    def main_layout():
        return html.Div([
            html.H3("Fragmentome explorer"),
            html.Div([
                dcc.Link('Admin', href='/admin', style={'padding': '10px', 'display': 'inline-block'}),
            ]),
        
        dcc.Loading(
            id="loading-icon",
            children=[html.Div(id='loading-status', children="Ready", style={'padding': '10px', 'color': 'green', 'fontWeight': 'bold'})],
            type="default"
        ),

        dcc.Upload(
            id='upload-alignment',
            children=html.Div(['Drag and Drop or ', html.A('Select BAM/SAM/CRAM')]),
            max_size=upload_max_size,
            style={
                'width': '20%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ),
        html.Div(id='upload-status', style={'padding': '0px 10px 10px 10px'}),

        html.Div([
            html.Div([
                html.Label("Color by:"),
                dcc.Dropdown(
                    id='color-dropdown',
                    options=[{'label': col, 'value': col} for col in color_columns],
                    value=color_columns[0] if color_columns else None,
                    clearable=True
                )
            ], style={'width': '20%', 'display': 'inline-block', 'padding': '10px'}),

            html.Div([
                html.Label("X axis:"),
                dcc.Dropdown(
                    id='x-dropdown',
                    options=[{'label': col, 'value': col} for col in numeric_columns],
                    value=default_x_col,
                    clearable=True
                )
            ], style={'width': '20%', 'display': 'inline-block', 'padding': '10px'}),

            html.Div([
                html.Label("Y axis:"),
                dcc.Dropdown(
                    id='y-dropdown',
                    options=[{'label': col, 'value': col} for col in numeric_columns],
                    value=default_y_col,
                    clearable=True
                )
            ], style={'width': '20%', 'display': 'inline-block', 'padding': '10px'}),

            html.Div([
                html.Label("Log scales:"),
                dcc.Checklist(
                    id='log-scales',
                    options=[
                        {'label': 'Log X', 'value': 'x'},
                        {'label': 'Log Y', 'value': 'y'},
                    ],
                    value=[],
                    inline=True
                )
            ], style={'width': '20%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                html.Label("Filter by:"),
                html.Button('Add filter', id='add-filter-btn', n_clicks=0, style={'marginLeft': '10px'}),
                html.Div(id='filter-container', style={'paddingTop': '10px'})
            ], style={'width': '40%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'}),
        ]),

        dcc.Loading(
            id="loading-scatter",
            children=[dcc.Graph(id='scatter-plot')],
            style={'padding': '10px', 'width': '100%', 'height': '100%'},
            type="graph"
        ),

        dcc.Dropdown(
            id='detail-columns-dropdown',
            options=[{'label': col, 'value': col} for col in color_columns],
            value=color_columns,
            multi=True,
            placeholder="Select columns to show in details"
        ),

        dcc.Checklist(
            id='sort-motifs-by-deviation',
            options=[{'label': 'Sort motif plots by |sample - cloud mean|', 'value': 'sort'}],
            value=[],
            inline=True,
            style={'padding': '10px'}
        ),

        html.Div(
            [
                html.Div(
                    id='point-details',
                    style={'flex': '1 1 0', 'minWidth': '360px', 'padding': '10px', 'overflowX': 'auto'}
                ),
                dcc.Loading(
                    id='loading-core-details',
                    type='default',
                    children=html.Div(
                        id='point-details-core',
                        style={'flex': '2 1 0', 'minWidth': '360px', 'padding': '10px'}
                    )
                ),
            ],
            style={'display': 'flex', 'flexDirection': 'row', 'flexWrap': 'nowrap', 'alignItems': 'flex-start', 'width': '100%'}
        ),

        html.Div(
            id='active-selection-table',
            style={'padding': '10px', 'width': '100%', 'overflowX': 'auto'}
        ),

        html.Div(
            [
                html.Button('Export table to TSV', id='export-active-table', n_clicks=0),
                dcc.Download(id='download-active-table-tsv')
            ],
            style={'padding': '0px 10px 10px 10px'}
        )
        ])

    def admin_layout(authenticated):
        if admin_password is None:
            return html.Div([
                html.H3('Admin'),
                dcc.Link('Back to explorer', href='/', style={'padding': '10px', 'display': 'inline-block'}),
                html.Div('Admin page disabled: set --admin-password or FRAGMENTOME_ADMIN_PASSWORD.', style={'padding': '10px'})
            ])

        if not authenticated:
            return html.Div([
                html.H3('Admin'),
                dcc.Link('Back to explorer', href='/', style={'padding': '10px', 'display': 'inline-block'}),
                html.Div([
                    html.Label('Admin password:'),
                    dcc.Input(id='admin-password-input', type='password', value='', style={'margin': '0 10px'}),
                    html.Button('Login', id='admin-login-btn', n_clicks=0),
                    html.Div(id='admin-login-status', style={'padding': '10px'})
                ], style={'padding': '10px'})
            ])

        type_options = ['UInt8', 'UInt16', 'UInt32', 'UInt64', 'Int8', 'Int16', 'Int32', 'Int64', 'Float32', 'Float64']
        dropdown_map = {
            'selected_type': {
                'options': [{'label': t, 'value': t} for t in type_options]
            }
        }

        return html.Div([
            html.H3('Admin'),
            html.Div([
                dcc.Link('Back to explorer', href='/', style={'padding': '10px', 'display': 'inline-block'}),
            ]),
            html.Div(
                f"Database: {db.get_sample_count(ch_client, db.CORE_TABLE)} core samples, {db.get_sample_count(ch_client, db.META_TABLE)} meta samples",
                style={'padding': '5px 10px', 'color': '#888', 'fontSize': '12px'}
            ),
            html.H4('Import Core TSV'),
            dcc.Upload(
                id='admin-upload-core-tsv',
                children=html.Div(['Drag and Drop or ', html.A('Select Core TSV')]),
                max_size=upload_max_size,
                style={
                    'width': '60%', 'height': '50px', 'lineHeight': '50px',
                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                    'textAlign': 'center', 'margin': '10px'
                },
                multiple=False
            ),
            html.Div(id='admin-core-status', style={'padding': '0px 10px 10px 10px'}),
            html.Div([
                dash_table.DataTable(
                    id='core-type-table',
                    columns=[
                        {'name': 'column', 'id': 'column'},
                        {'name': 'inferred_type', 'id': 'inferred_type'},
                        {'name': 'selected_type', 'id': 'selected_type', 'presentation': 'dropdown'},
                    ],
                    data=[],
                    dropdown=dropdown_map,
                    page_size=25,
                    sort_action='native',
                    filter_action='native',
                    style_table={'height': '420px', 'overflowY': 'auto', 'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'fontFamily': 'monospace', 'fontSize': '12px', 'whiteSpace': 'normal', 'height': 'auto'},
                ),
            ], style={'padding': '10px'}),
            html.Div([
                dcc.Checklist(
                    id='recreate-core-table',
                    options=[{'label': 'Recreate core table (drops existing core data)', 'value': 'recreate'}],
                    value=['recreate'],
                    inline=True
                ),
                html.Button('Import core into ClickHouse', id='admin-import-core-btn', n_clicks=0,
                            style={'marginLeft': '10px'}),
            ], style={'padding': '10px'}),
            html.Div(id='admin-import-core-status', style={'padding': '0px 10px 10px 10px'}),
            html.H4('Import Meta TSV'),
            dcc.Upload(
                id='admin-upload-meta-tsv',
                children=html.Div(['Drag and Drop or ', html.A('Select Meta TSV')]),
                max_size=upload_max_size,
                style={
                    'width': '60%', 'height': '50px', 'lineHeight': '50px',
                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                    'textAlign': 'center', 'margin': '10px'
                },
                multiple=False
            ),
            html.Div(id='admin-meta-status', style={'padding': '0px 10px 10px 10px'}),
            html.H4('Database Statistics'),
            html.Div([
                html.Button('Update Core Stats', id='admin-update-stats-btn', n_clicks=0,
                            style={'backgroundColor': '#28a745', 'color': 'white', 'border': 'none',
                                   'padding': '8px 16px', 'borderRadius': '4px', 'cursor': 'pointer', 'marginRight': '10px'}),
                html.Button('Update Meta Features', id='admin-update-features-btn', n_clicks=0,
                            style={'backgroundColor': '#17a2b8', 'color': 'white', 'border': 'none',
                                   'padding': '8px 16px', 'borderRadius': '4px', 'cursor': 'pointer', 'marginRight': '10px'}),
            ], style={'padding': '10px'}),
            html.Div(id='admin-update-stats-status', style={'padding': '0px 10px 10px 10px'}),
            html.Div([
                html.Button('Clear database (core+meta)', id='admin-clear-db-btn', n_clicks=0,
                            style={'backgroundColor': '#dc3545', 'color': 'white', 'border': 'none',
                                   'padding': '8px 16px', 'borderRadius': '4px', 'cursor': 'pointer'}),
                html.Div(id='admin-clear-db-status', style={'display': 'inline-block', 'padding': '0 10px', 'color': '#555'})
            ], style={'padding': '10px'}),
        ])

    @app.callback(
        Output('page-content', 'children'),
        Input('url', 'pathname'),
        Input('admin-auth', 'data')
    )
    def render_page(pathname, authenticated):
        log.info('Navigation: pathname=%s authenticated=%s', pathname, bool(authenticated))
        if pathname == '/admin':
            return admin_layout(bool(authenticated))
        return main_layout()

    @app.callback(
        Output('filter-indices', 'data'),
        Input('add-filter-btn', 'n_clicks'),
        Input({'type': 'remove-filter-btn', 'index': dash.ALL}, 'n_clicks'),
        State('filter-indices', 'data'),
        prevent_initial_call=True
    )
    def update_filter_indices(add_clicks, remove_clicks, indices):
        ctx = dash.callback_context
        if not ctx.triggered:
            return indices

        trig = ctx.triggered[0]['prop_id']
        indices = list(indices or [])

        if trig == 'add-filter-btn.n_clicks':
            if not add_clicks:
                return indices
            next_idx = (max(indices) + 1) if indices else 0
            indices.append(next_idx)
            log.info('Added filter row index=%s', next_idx)
            return indices

        # Remove button triggered
        try:
            trig_id = ctx.triggered_id
            if isinstance(trig_id, dict) and trig_id.get('type') == 'remove-filter-btn':
                if not remove_clicks or not any(remove_clicks):
                    return indices
                rm = trig_id.get('index')
                indices = [i for i in indices if i != rm]
                log.info('Removed filter row index=%s', rm)
                return indices
        except Exception:
            return indices

        return indices

    @app.callback(
        Output('filter-container', 'children'),
        Input('filter-indices', 'data')
    )
    def render_filters(indices):
        indices = list(indices or [])
        children = []
        for i in indices:
            children.append(
                html.Div([
                    dcc.Dropdown(
                        id={'type': 'filter-column', 'index': i},
                        options=[{'label': col, 'value': col} for col in color_columns],
                        value=None,
                        clearable=True,
                        placeholder='Column',
                        style={'width': '35%', 'display': 'inline-block'}
                    ),
                    dcc.Dropdown(
                        id={'type': 'filter-op', 'index': i},
                        options=[
                            {'label': 'Range (numeric)', 'value': 'range'},
                            {'label': 'Matches (regex)', 'value': '=='},
                            {'label': 'Does not match (regex)', 'value': '!='},
                        ],
                        value='range',
                        clearable=False,
                        style={'width': '25%', 'display': 'inline-block', 'marginLeft': '8px'}
                    ),
                    dcc.Input(
                        id={'type': 'filter-lo', 'index': i},
                        type='number',
                        placeholder='lo',
                        value=None,
                        style={'width': '12%', 'display': 'inline-block', 'marginLeft': '8px'}
                    ),
                    dcc.Input(
                        id={'type': 'filter-hi', 'index': i},
                        type='number',
                        placeholder='hi',
                        value=None,
                        style={'width': '12%', 'display': 'inline-block', 'marginLeft': '8px'}
                    ),
                    dcc.Input(
                        id={'type': 'filter-expr', 'index': i},
                        type='text',
                        placeholder='expr (regex)',
                        value='',
                        style={'width': '30%', 'display': 'inline-block', 'marginLeft': '8px'}
                    ),
                    html.Button('Remove', id={'type': 'remove-filter-btn', 'index': i}, n_clicks=0, style={'marginLeft': '8px'}),
                ], style={'padding': '4px 0'})
            )
        return children

    @app.callback(
        Output('filters-store', 'data'),
        Input({'type': 'filter-column', 'index': dash.ALL}, 'value'),
        Input({'type': 'filter-op', 'index': dash.ALL}, 'value'),
        Input({'type': 'filter-lo', 'index': dash.ALL}, 'value'),
        Input({'type': 'filter-hi', 'index': dash.ALL}, 'value'),
        Input({'type': 'filter-expr', 'index': dash.ALL}, 'value'),
        prevent_initial_call=True
    )
    def collect_filters(cols, ops, los, his, exprs):
        filters = []
        n = max(len(cols), len(ops), len(los), len(his), len(exprs)) if cols is not None else 0
        for i in range(n):
            col = cols[i] if i < len(cols) else None
            op = ops[i] if i < len(ops) else None
            lo = los[i] if i < len(los) else None
            hi = his[i] if i < len(his) else None
            expr = exprs[i] if i < len(exprs) else None
            if not col or not op:
                continue
            if op == 'range':
                if lo is None or hi is None:
                    continue
            elif op in ('==', '!='):
                if expr is None or not str(expr).strip():
                    continue
            filters.append({'column': col, 'op': op, 'lo': lo, 'hi': hi, 'expr': expr})
        log.debug('Collected %s active filters', len(filters))
        return filters

    @app.callback(
        Output('cloud-ids-core', 'data'),
        Input('filters-store', 'data')
    )
    def compute_cloud_ids_core(filters):
        # Precompute core-eligible cloud IDs for current filter state.
        t_cloud = time.time()
        
        # Use fresh client to avoid concurrent query issues
        fresh_client = db.get_fresh_client(
            host=getattr(args, 'ch_host', 'localhost'),
            port=getattr(args, 'ch_port', 8123)
        )
        
        filtered_sample_ids = _apply_filters(filters, client=fresh_client)
        
        # Get intersection with core table samples
        core_samples_result = fresh_client.query(f'SELECT DISTINCT sample_id FROM {db.DATABASE_NAME}.{db.CORE_TABLE}')
        core_sample_ids = {row[0] for row in core_samples_result.result_rows}
        
        cloud_ids_core = [sid for sid in filtered_sample_ids if sid in core_sample_ids]

        # Cap to keep core-details responsive
        max_cloud = 5000
        if len(cloud_ids_core) > max_cloud:
            cloud_ids_core = cloud_ids_core[:max_cloud]

        log.info('Computed cloud_ids_core in %.2fs (filtered=%s, cloud_core=%s)', time.time() - t_cloud, len(filtered_sample_ids), len(cloud_ids_core))
        return cloud_ids_core

    @app.callback(
        Output('uploaded-point', 'data'),
        Output('upload-status', 'children'),
        Output('x-dropdown', 'value'),
        Output('y-dropdown', 'value'),
        Output('loading-status', 'children'),
        Input('upload-alignment', 'contents'),
        State('upload-alignment', 'filename'),
        State('x-dropdown', 'value'),
        State('y-dropdown', 'value')
    )
    def handle_alignment_upload(contents, filename, current_x, current_y):
        if contents is None:
            log.info('User cancelled alignment upload')
            return None, '', current_x, current_y, 'Ready'

        if not isinstance(contents, str):
            return None, f'Upload failed: unexpected contents type {type(contents)}.', current_x, current_y, 'Error: Invalid content type'

        # Show processing status before validation
        processing_status = '🔄 Processing uploaded file...'

        if mapping is None:
            return None, 'Upload disabled: start the app with --mapping <mapping.pkl> so I can compute x/y coordinates.', current_x, current_y, 'Error: Mapping not loaded'

        if args.reference is None:
            return None, 'Upload disabled: provide --reference <fasta> (required for motif features).', current_x, current_y, 'Error: Reference not provided'

        try:
            header, b64data = contents.split(',', 1)
            data = base64.b64decode(b64data)
        except Exception:
            return None, 'Upload failed: could not decode file contents.', current_x, current_y, processing_status

        ext = ''
        if filename:
            ext = os.path.splitext(filename)[1]
        if ext.lower() not in ['.bam', '.sam', '.cram']:
            return None, f'Upload failed: unsupported file type {ext}. Please upload .bam, .sam, or .cram.', current_x, current_y, processing_status

        log.info('Processing alignment upload: %s', filename)

        try:
            b64data = contents.split(',', 1)[1] if ',' in contents else contents
            b64data = b64data.strip().replace('\n', '').replace('\r', '')
            pad = (-len(b64data)) % 4
            if pad:
                b64data += '=' * pad

            data = base64.b64decode(b64data)
        except Exception as e:
            return None, f'Upload failed: could not decode file contents ({type(e).__name__}: {str(e)}).', current_x, current_y, processing_status

        try:
            uploads_dir = os.path.join(os.getcwd(), 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)

            base = os.path.basename(filename) if filename else f'upload{ext}'
            base = base.replace('\x00', '')
            base = base.replace('/', '_').replace('\\', '_')
            if not base.lower().endswith(ext.lower()):
                base = base + ext

            out_path = os.path.join(uploads_dir, base)
            if os.path.exists(out_path):
                root, e = os.path.splitext(base)
                out_path = os.path.join(uploads_dir, f"{root}.{uuid.uuid4().hex[:8]}{e}")

            with open(out_path, 'wb') as f_out:
                f_out.write(data)

            log.info('Uploaded file saved to: %s', out_path)
            
            # Update status to show processing
            log.info('Computing features for uploaded file (k=%s)', mapping_k)
            
            upload_args = copy.copy(args)
            upload_args.samfiles = [out_path]
            upload_args.bamlist = None
            upload_args.k = int(mapping_k)
            upload_args.norm = 'rpx'
            upload_args.x = 1000000
            upload_args.exclflag = 3852
            upload_args.mapqual = 60
            upload_args.purpyr = False
            upload_args.uselexsmallest = False
            upload_args.useref = False
            upload_args.insertissize = True
            upload_args.lower = 0
            upload_args.upper = 1000
            
            Xfszd = np.array(fszd.fszd(upload_args, cmdline=False))
            log.debug('Xfszd shape: %s', getattr(Xfszd, 'shape', None))
            Xcsm = np.array(csm.cleavesitemotifs(upload_args, cmdline=False))
            log.debug('Xcsm shape: %s', getattr(Xcsm, 'shape', None))
            Xsem = np.array(fpends._5pends(upload_args, cmdline=False))
            log.debug('Xsem shape: %s', getattr(Xsem, 'shape', None))
            f = np.concatenate((Xfszd, Xcsm, Xsem), axis=1)
            
            log.info('Computed features for %s (shape=%s); running reducer transform', filename, getattr(f, 'shape', None))
            
            
            fp = reducer.transform(f)
            log.debug('Reduced features shape: %s', getattr(fp, 'shape', None))

            x_new = float(fp[0, 0])
            y_new = float(fp[0, 1])
            label = filename if filename else os.path.basename(out_path)

            log.info('Finished processing uploaded alignment: %s', filename)
            
            # Insert results into ClickHouse
            try:
                sample_id = db.insert_alignment_results(ch_client, filename, f, x_new, y_new)
                if sample_id:
                    log.info('Inserted alignment results into ClickHouse with sample_id: %s', sample_id)
                else:
                    log.warning('Failed to insert alignment results into ClickHouse')
            except Exception as e:
                log.exception('Failed to insert alignment results into ClickHouse')
            
            log.info('Alignment processed: %s (umap1=%.3f, umap2=%.3f)', label, x_new, y_new)
            return {
                'umap1': x_new,
                'umap2': y_new,
                'label': label,
            }, f'Processed: {label} (umap1={x_new:.3f}, umap2={y_new:.3f})', 'umap1', 'umap2', f'✓ Processed: {label}'
        except Exception as e:
            log.exception('Alignment upload failed for %s', filename)
            return None, f'Upload failed: {str(e)}', current_x, current_y, f'Error: {str(e)}'
        finally:
            pass

    @app.callback(
        Output('admin-auth', 'data'),
        Output('admin-login-status', 'children'),
        Input('admin-login-btn', 'n_clicks'),
        State('admin-password-input', 'value'),
        prevent_initial_call=True
    )
    def admin_login(n_clicks, password_input):
        if not n_clicks:
            return dash.no_update, ''
        if admin_password is None:
            return False, 'Admin page disabled.'
        if password_input == admin_password:
            log.info('Admin login succeeded')
            return True, '✓ Logged in.'
        log.warning('Admin login failed')
        return False, '✗ Invalid password.'

    @app.callback(
        Output('core-tsv-bytes', 'data'),
        Output('core-column-types', 'data'),
        Output('core-type-table', 'data'),
        Output('admin-core-status', 'children'),
        Input('admin-upload-core-tsv', 'contents'),
        State('admin-upload-core-tsv', 'filename'),
        prevent_initial_call=True
    )
    def admin_handle_core_tsv(contents, filename):
        if contents is None:
            return None, None, [], ''
        try:
            log.info('Admin uploaded core TSV: %s', filename)
            header, b64data = contents.split(',', 1)
            raw = base64.b64decode(b64data)
            df_preview = pd.read_csv(io.StringIO(raw.decode('utf-8')), sep='\t', index_col=0)
            inferred = db.infer_core_column_types(df_preview, sample_rows=100)
            table_data = []
            for col in df_preview.columns:
                c = str(col)
                it = inferred.get(c, 'UInt32')
                table_data.append({'column': c, 'inferred_type': it, 'selected_type': it})
            return base64.b64encode(raw).decode('ascii'), inferred, table_data, f'✓ Loaded {filename}. Review types and click import.'
        except Exception as e:
            log.exception('Admin failed to read core TSV: %s', filename)
            return None, None, [], f'✗ Failed to read core TSV: {str(e)}'

    @app.callback(
        Output('admin-import-core-status', 'children'),
        Input('admin-import-core-btn', 'n_clicks'),
        State('core-tsv-bytes', 'data'),
        State('core-type-table', 'data'),
        State('recreate-core-table', 'value'),
        prevent_initial_call=True
    )
    def admin_import_core(n_clicks, core_bytes_b64, type_table_data, recreate_value):
        if not n_clicks:
            return ''
        if not core_bytes_b64:
            return '✗ Upload a core TSV first.'
        try:
            log.info('Admin requested core import (recreate=%s)', bool(recreate_value) and ('recreate' in recreate_value))
            raw = base64.b64decode(core_bytes_b64)
            column_types = {}
            if type_table_data:
                for row in type_table_data:
                    if row.get('column') and row.get('selected_type'):
                        column_types[str(row['column'])] = str(row['selected_type'])

            recreate = bool(recreate_value) and ('recreate' in recreate_value)
            n = db.upload_core_from_bytes(ch_client, raw, column_types=column_types, recreate_table=recreate)
            
            # Update core column groups after import
            nonlocal core_x, core_col_names, csm_col_names, csm_x, p5_col_names, p5_x
            fresh_client = db.get_fresh_client(
                host=getattr(args, 'ch_host', 'localhost'),
                port=getattr(args, 'ch_port', 8123)
            )
            columns_result = fresh_client.query(
                f"SELECT name FROM system.columns WHERE database='{db.DATABASE_NAME}' AND table='{db.CORE_TABLE}' AND name != 'sample_id' ORDER BY position"
            )
            if columns_result.result_rows:
                core_columns = [row[0] for row in columns_result.result_rows]
                # Re-extract core, csm, and 5p column groups
                core_x, core_col_names, csm_col_names, csm_x, p5_col_names, p5_x = [], [], [], [], [], []
                for c in core_columns:
                    try:
                        ci = int(c)
                    except (TypeError, ValueError):
                        continue
                    if 1 <= ci <= 1000:
                        core_x.append(ci)
                        core_col_names.append(c)
                
                csm_col_names = [c for c in core_columns if str(c).endswith('-csm')]
                csm_x = [str(c)[:-4] for c in csm_col_names]
                p5_col_names = [c for c in core_columns if str(c).endswith('-5p')]
                p5_x = [str(c)[:-3] for c in p5_col_names]
            
            log.info('Admin core import complete: %s samples', n)
            return f'✓ Imported {n} core samples into ClickHouse.'
        except Exception as e:
            log.exception('Admin core import failed')
            return f'✗ Core import failed: {str(e)}'

    @app.callback(
        Output('admin-meta-status', 'children'),
        Input('admin-upload-meta-tsv', 'contents'),
        State('admin-upload-meta-tsv', 'filename'),
        prevent_initial_call=True
    )
    def admin_handle_meta_tsv(contents, filename):
        if contents is None:
            return ''
        try:
            log.info('Admin uploaded meta TSV: %s', filename)
            header, b64data = contents.split(',', 1)
            raw = base64.b64decode(b64data)
            n = db.upload_meta_from_bytes(ch_client, raw)
            
            # Update meta features after import
            nonlocal color_columns, numeric_columns
            meta_features = db.get_meta_features(ch_client)
            color_columns = meta_features
            numeric_columns = meta_features  # Simplified - will be refined per-query
            
            log.info('Admin meta import complete: %s rows', n)
            return f'✓ Imported {n} meta rows into ClickHouse.'
        except Exception as e:
            log.exception('Admin meta import failed', filename)
            return f'✗ Meta import failed: {str(e)}'

    @app.callback(
        Output('admin-clear-db-status', 'children'),
        Input('admin-clear-db-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def admin_clear_db(n_clicks):
        nonlocal color_columns, numeric_columns
        nonlocal core_x, core_col_names, csm_col_names, csm_x, p5_col_names, p5_x
        if not n_clicks:
            return ''
        try:
            log.warning('Admin requested database clear')
            db.drop_tables(ch_client)
            color_columns = []
            numeric_columns = []
            core_x, core_col_names, csm_col_names, csm_x, p5_col_names, p5_x = [], [], [], [], [], []
            return '✓ Database cleared.'
        except Exception as e:
            log.exception('Admin clear database failed')
            return f'✗ Clear failed: {str(e)}'

    @app.callback(
        Output('admin-update-stats-status', 'children'),
        Input('admin-update-stats-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def admin_update_core_stats(n_clicks):
        if not n_clicks:
            return ''
        try:
            log.info('Admin requested core stats update')
            n_stats = db.update_core_stats(ch_client)
            log.info('Admin core stats update complete: %s features', n_stats)
            return f'✓ Updated core statistics for {n_stats} features.'
        except Exception as e:
            log.exception('Admin core stats update failed')
            return f'✗ Core stats update failed: {str(e)}'

    @app.callback(
        Output('admin-update-stats-status', 'children', allow_duplicate=True),
        Input('admin-update-features-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def admin_update_meta_features(n_clicks):
        if not n_clicks:
            return ''
        try:
            log.info('Admin requested meta features update')
            nonlocal color_columns, numeric_columns
            n_features = db.update_meta_features(ch_client)
            meta_features = db.get_meta_features(ch_client)
            color_columns = meta_features
            numeric_columns = meta_features
            log.info('Admin meta features update complete: %s features', n_features)
            return f'✓ Updated meta features for {n_features} features.'
        except Exception as e:
            log.exception('Admin meta features update failed')
            return f'✗ Meta features update failed: {str(e)}'

    @app.callback(
        Output('selected-sample', 'data'),
        Input('scatter-plot', 'clickData')
    )
    def update_selected_sample(clickData):
        if clickData is None or 'points' not in clickData or not clickData['points']:
            return dash.no_update

        point = clickData['points'][0]
        if 'customdata' not in point or not point['customdata']:
            return dash.no_update

        sample_id = str(point['customdata'][0])
        log.info('User selected sample: %s', sample_id)
        return sample_id

    @app.callback(
        Output('scatter-plot', 'figure'),
        Input('x-dropdown', 'value'),
        Input('y-dropdown', 'value'),
        Input('color-dropdown', 'value'),
        Input('log-scales', 'value'),
        Input('filters-store', 'data'),
        Input('uploaded-point', 'data')
    )

    def update_figure(selected_x, selected_y, selected_color, log_scales, filters, uploaded_point):
        t_fig = time.time()
        log.info('Figure update: x=%s y=%s color=%s filters=%s uploaded=%s', selected_x, selected_y, selected_color, len(filters or []), bool(uploaded_point))

        # Use fresh client to avoid concurrent query issues
        fresh_client = db.get_fresh_client(
            host=getattr(args, 'ch_host', 'localhost'),
            port=getattr(args, 'ch_port', 8123)
        )
        
        # Get filtered sample IDs
        filtered_sample_ids = _apply_filters(filters, client=fresh_client)
        
        if not filtered_sample_ids:
            fig = go.Figure()
            fig.update_layout(template='plotly_white', title='No data matches current filters')
            log.info('Figure update finished in %.2fs (no data)', time.time() - t_fig)
            return fig

        # Determine which features we need to load
        features_needed = []
        if selected_x:
            features_needed.append(selected_x)
        if selected_y:
            features_needed.append(selected_y)
        if selected_color:
            features_needed.append(selected_color)
        
        # Add _sample_id for custom data
        features_needed.append('_sample_id')
        
        # Lazily load only the data we need
        df_filtered = db.lazy_load_meta_samples(fresh_client, sample_ids=filtered_sample_ids, features=features_needed)

        if not df_filtered.empty and '_sample_id' not in df_filtered.columns:
            df_filtered = df_filtered.copy()
            df_filtered['_sample_id'] = df_filtered.index.astype(str)
        
        if df_filtered.empty:
            fig = go.Figure()
            fig.update_layout(template='plotly_white', title='No data available for selected features')
            log.info('Figure update finished in %.2fs (empty data)', time.time() - t_fig)
            return fig

        if selected_x is None and selected_y is None:
            fig = go.Figure()
            fig.update_layout(template='plotly_white', title='Select X and/or Y to plot')
            log.info('Figure update finished in %.2fs (no axes)', time.time() - t_fig)
            return fig
        
        color_is_categorical = False
        if selected_color is not None and selected_color in df_filtered.columns:
            color_is_categorical = not pd.api.types.is_numeric_dtype(df_filtered[selected_color])

        if selected_x is not None and selected_y is None:
            if color_is_categorical:
                fig = px.histogram(
                    df_filtered,
                    y=selected_x,
                    color=selected_color,
                    histnorm='probability',
                    template='plotly_white',
                    title=f"Histogram of {selected_x} grouped by {selected_color} ({len(df_filtered)} points)",
                    orientation='h',
                    opacity=0.7
                )
                fig.update_layout(barmode='overlay')
            else:
                fig = px.histogram(
                    df_filtered,
                    y=selected_x,
                    template='plotly_white',
                    title=f"Histogram of {selected_x} ({len(df_filtered)} points)",
                    orientation='h'
                )
            fig.update_layout(uirevision=f"hx|{selected_x}|{selected_color}|{len(filters or [])}")
            fig.update_layout(clickmode='event+select')
            if log_scales and 'x' in log_scales:
                fig.update_xaxes(type='log')
            if log_scales and 'y' in log_scales:
                fig.update_yaxes(type='log')
            log.info('Figure update finished in %.2fs (histogram)', time.time() - t_fig)
            return fig

        if selected_y is not None and selected_x is None:
            if color_is_categorical:
                fig = px.histogram(
                    df_filtered,
                    x=selected_y,
                    color=selected_color,
                    histnorm='probability',
                    template='plotly_white',
                    title=f"Histogram of {selected_y} grouped by {selected_color} ({len(df_filtered)} points)",
                    opacity=0.7
                )
                fig.update_layout(barmode='overlay')
            else:
                fig = px.histogram(
                    df_filtered,
                    x=selected_y,
                    template='plotly_white',
                    title=f"Histogram of {selected_y} ({len(df_filtered)} points)"
                )
            fig.update_layout(uirevision=f"vy|{selected_y}|{selected_color}|{len(filters or [])}")
            fig.update_layout(clickmode='event+select')
            if log_scales and 'x' in log_scales:
                fig.update_xaxes(type='log')
            if log_scales and 'y' in log_scales:
                fig.update_yaxes(type='log')
            return fig

        df_sorted = df_filtered
        if selected_color is not None and selected_color in df_filtered.columns:
            is_categorical = not pd.api.types.is_numeric_dtype(df_filtered[selected_color]) or \
                             (pd.api.types.is_numeric_dtype(df_filtered[selected_color]) and 
                              len(df_filtered[selected_color].dropna().unique()) <= 25 and
                              all(val.is_integer() if isinstance(val, float) else True 
                                  for val in df_filtered[selected_color].dropna().unique()))

            if is_categorical:
                freq = df_filtered[selected_color].value_counts()
                df_sorted = df_filtered.copy()
                df_sorted[selected_color] = pd.Categorical(
                    df_sorted[selected_color],
                    categories=freq.index.tolist(),
                    ordered=True
                )
                df_sorted = df_sorted.sort_values(selected_color)

            if pd.api.types.is_numeric_dtype(df_filtered[selected_color]) and not is_categorical:
                fig = px.scatter(
                    df_sorted,
                    x=selected_x,
                    y=selected_y,
                    color=selected_color,
                    color_continuous_scale='Viridis',
                    template="plotly_white",
                    hover_data=[selected_x, selected_y],
                    custom_data=['_sample_id'],
                    labels={selected_color: selected_color},
                    title=f"Scatter plot colored by {selected_color} ({len(df_filtered)} points)",
                )
            else:
                fig = px.scatter(
                    df_sorted,
                    x=selected_x,
                    y=selected_y,
                    color=selected_color,
                    template="plotly_white",
                    hover_data=[selected_x, selected_y],
                    custom_data=['_sample_id'],
                    labels={selected_color: selected_color},
                    title=f"Scatter plot colored by {selected_color} ({len(df_filtered)} points)",
                )
        else:
            fig = px.scatter(
                df_sorted,
                x=selected_x,
                y=selected_y,
                template="plotly_white",
                hover_data=[selected_x, selected_y],
                custom_data=['_sample_id'],
                title=f"Scatter plot ({len(df_filtered)} points)",
            )

        # fig.update_yaxes(scaleanchor='x', scaleratio=1)

        uirev = f"{selected_x}|{selected_y}|{selected_color}|{len(filters or [])}"
        fig.update_layout(clickmode='event+select', uirevision=uirev)

        if log_scales and 'x' in log_scales:
            fig.update_xaxes(type='log')
        if log_scales and 'y' in log_scales:
            fig.update_yaxes(type='log')

        if selected_x == 'umap1' and selected_y == 'umap2' and uploaded_point is not None and 'umap1' in uploaded_point and 'umap2' in uploaded_point:
            fig.add_trace(
                go.Scatter(
                    x=[uploaded_point['umap1']],
                    y=[uploaded_point['umap2']],
                    mode='markers',
                    marker=dict(size=18, symbol='star', color='red', line=dict(color='darkred', width=2)),
                    name='Uploaded sample',
                    hovertemplate=f"{uploaded_point.get('label','Uploaded')}<br>umap1=%{{x:.3f}}<br>umap2=%{{y:.3f}}<extra></extra>",
                )
            )
            # Ensure the uploaded point is always on top
            fig.data[-1].update(showlegend=True)

        return fig

    @app.callback(
        Output('active-selection-table', 'children'),
        Input('scatter-plot', 'selectedData'),
        Input('x-dropdown', 'value'),
        Input('y-dropdown', 'value'),
        Input('filters-store', 'data')
    )
    def update_active_selection_table(selected_data, selected_x, selected_y, filters):
        t_sel = time.time()
        log.info('Active-selection table update: has_selection=%s', bool(selected_data and isinstance(selected_data, dict) and selected_data.get('points')))
        
        # Get filtered sample IDs
        filtered_sample_ids = _apply_filters(filters)
        
        if selected_x is None and selected_y is None:
            return ''

        # Apply selection if present
        if selected_data is not None and isinstance(selected_data, dict) and 'points' in selected_data and selected_data['points']:
            selected_sample_ids = []
            for p in selected_data['points']:
                cd = p.get('customdata')
                if cd:
                    selected_sample_ids.append(str(cd[0]))
            if selected_sample_ids:
                filtered_sample_ids = [sid for sid in filtered_sample_ids if sid in selected_sample_ids]
        
        if not filtered_sample_ids:
            return dash_table.DataTable(data=[], columns=[], page_size=50)
        
        # Load all meta features for the selected samples
        fresh_client = db.get_fresh_client(
            host=getattr(args, 'ch_host', 'localhost'),
            port=getattr(args, 'ch_port', 8123)
        )
        df_filtered = db.lazy_load_meta_samples(fresh_client, sample_ids=filtered_sample_ids)
        
        if df_filtered.empty:
            return dash_table.DataTable(data=[], columns=[], page_size=50)

        df_head = df_filtered.head(50)#.reset_index()
        cols = [{'name': c, 'id': c} for c in df_head.columns]

        log.info('Active-selection table update finished in %.2fs (rows=%s)', time.time() - t_sel, len(df_head))

        return dash_table.DataTable(
            data=df_head.to_dict('records'),
            columns=cols,
            page_size=50,
            sort_action='native',
            filter_action='none',
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'fontFamily': 'monospace', 'fontSize': '12px', 'whiteSpace': 'normal', 'height': 'auto'},
        )

    @app.callback(
        Output('download-active-table-tsv', 'data'),
        Input('export-active-table', 'n_clicks'),
        State('scatter-plot', 'selectedData'),
        State('x-dropdown', 'value'),
        State('y-dropdown', 'value'),
        State('filters-store', 'data'),
        prevent_initial_call=True
    )
    def export_active_selection_table(n_clicks, selected_data, selected_x, selected_y, filters):
        if n_clicks:
            log.info('User requested TSV export (n_clicks=%s)', n_clicks)
        
        # Get filtered sample IDs
        filtered_sample_ids = _apply_filters(filters)

        if selected_x is None and selected_y is None:
            return None

        # Apply selection if present
        if selected_data is not None and isinstance(selected_data, dict) and 'points' in selected_data and selected_data['points']:
            selected_sample_ids = []
            for p in selected_data['points']:
                cd = p.get('customdata')
                if cd:
                    selected_sample_ids.append(str(cd[0]))
            if selected_sample_ids:
                filtered_sample_ids = [sid for sid in filtered_sample_ids if sid in selected_sample_ids]
        
        if not filtered_sample_ids:
            return None
        
        # Load all meta features for the selected samples
        fresh_client = db.get_fresh_client(
            host=getattr(args, 'ch_host', 'localhost'),
            port=getattr(args, 'ch_port', 8123)
        )
        df_filtered = db.lazy_load_meta_samples(fresh_client, sample_ids=filtered_sample_ids)
        
        if df_filtered.empty:
            return None

        df_out = df_filtered.head(50) #.reset_index()
        buf = io.StringIO()
        df_out.to_csv(buf, sep='\t', index=False)
        return dict(content=buf.getvalue(), filename='active_selection.tsv', type='text/tab-separated-values')

    @app.callback(
        Output('point-details', 'children'),
        Input('scatter-plot', 'clickData'),
        Input('selected-sample', 'data'),
        Input('detail-columns-dropdown', 'value'),
        Input('filters-store', 'data')
    )
    
    def display_point_details(clickData, selected_sample, selected_columns, filters):
        log.debug('Point details update')
        sample_id = None
        if clickData is not None and 'points' in clickData and clickData['points']:
            point = clickData['points'][0]
            if 'customdata' in point and point['customdata']:
                sample_id = str(point['customdata'][0])

        if sample_id is None and selected_sample is not None:
            sample_id = str(selected_sample)

        if sample_id is None:
            return "Click a point in the scatter plot to see its values here."

        # Load meta data for this specific sample
        fresh_client = db.get_fresh_client(
            host=getattr(args, 'ch_host', 'localhost'),
            port=getattr(args, 'ch_port', 8123)
        )
        df_sample = db.lazy_load_meta_samples(fresh_client, sample_ids=[sample_id])
        
        if df_sample.empty:
            return "Could not determine selected point (sample id not found in metadata table)."

        row = df_sample.iloc[0]  # Get the first (and only) row
        log.debug('Point details for %s (row_shape=%s)', sample_id, getattr(row, 'shape', None))
        
        # Use selected columns, fallback to all available columns if none selected
        columns_to_show = selected_columns if selected_columns else list(df_sample.columns)
        
        # Filter to only show columns that exist
        columns_to_show = [col for col in columns_to_show if col in df_sample.columns]

        return html.Table([
            html.Thead(html.Tr([
                html.Th("Field"),
                html.Th("Value")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(str(col)),
                    html.Td(str(row[col]))
                ]) for col in columns_to_show
            ])
        ])

    @app.callback(
        Output('point-details-core', 'children'),
        Input('scatter-plot', 'clickData'),
        Input('selected-sample', 'data'),
        Input('sort-motifs-by-deviation', 'value')
    )
    def display_point_details_core(clickData, selected_sample, sort_motifs_value):
        t_core = time.time()
        log.debug('Core details update (clickData_present=%s)', bool(clickData))
        sample_id = None
        if clickData is not None and 'points' in clickData and clickData['points']:
            point = clickData['points'][0]
            if 'customdata' in point and point['customdata']:
                sample_id = str(point['customdata'][0])

        if sample_id is None and selected_sample is not None:
            sample_id = str(selected_sample)

        if sample_id is None:
            return ""

        log.debug('Core details sample_id=%s', sample_id)

        # Check if sample exists in core table
        fresh_client = db.get_fresh_client(
            host=getattr(args, 'ch_host', 'localhost'),
            port=getattr(args, 'ch_port', 8123)
        )
        core_samples_result = fresh_client.query(f"SELECT sample_id FROM {db.DATABASE_NAME}.{db.CORE_TABLE} WHERE sample_id = '{sample_id}'")
        if not core_samples_result.result_rows:
            log.warning('Selected sample %s not present in core table', sample_id)
            return f"Selected sample {sample_id} is not present in the core feature table."

        # Load global core feature statistics (mean/std) from core_stats table.
        df_core_stats = db.get_core_stats(fresh_client)
        if not df_core_stats.empty and 'feature_name' in df_core_stats.columns:
            df_core_stats = df_core_stats.set_index('feature_name')
        else:
            df_core_stats = pd.DataFrame()

        log.info('Core details loaded core_stats (n=%s) (prep_total=%.2fs)', len(df_core_stats) if hasattr(df_core_stats, '__len__') else 0, time.time() - t_core)

        sort_motifs = bool(sort_motifs_value) and ('sort' in sort_motifs_value)

        def make_feature_comparison_figure(title, x_vals, y_sample, mean_vals=None, std_vals=None, sort_x_by_deviation=False, x_is_categorical=False):
            log.debug('make_feature_comparison_figure: %s', title)
            fig_local = go.Figure()

            x_vals_out = list(x_vals)
            y_sample_out = y_sample.copy()

            if mean_vals is not None and std_vals is not None:
                mean_vals_local = mean_vals
                std_vals_local = std_vals

                if sort_x_by_deviation:
                    dev = (y_sample_out - mean_vals_local).abs()
                    order = dev.sort_values(ascending=False).index.tolist()
                    x_by_col = dict(zip(list(y_sample.index), list(x_vals)))
                    x_vals_out = [x_by_col.get(col, str(col)) for col in order]
                    y_sample_out = y_sample_out.loc[order]
                    mean_vals_local = mean_vals_local.loc[order]
                    std_vals_local = std_vals_local.loc[order]

                mean_vals_arr = mean_vals_local.values
                std_vals_arr = std_vals_local.values

                if len(mean_vals_arr) == len(x_vals_out) and len(std_vals_arr) == len(x_vals_out):
                    upper = mean_vals_arr + std_vals_arr
                    lower = mean_vals_arr - std_vals_arr
                    fig_local.add_trace(
                        go.Scatter(
                            x=list(x_vals_out) + list(x_vals_out)[::-1],
                            y=list(upper) + list(lower[::-1]),
                            fill='toself',
                            fillcolor='rgba(99, 110, 250, 0.15)',
                            line={'color': 'rgba(255,255,255,0)'},
                            hoverinfo='skip',
                            name='Mean ± 1 SD'
                        )
                    )
                    fig_local.add_trace(
                        go.Scatter(
                            x=x_vals_out,
                            y=mean_vals_arr,
                            mode='lines',
                            line={'color': 'rgba(99, 110, 250, 0.9)', 'width': 2},
                            name='Mean'
                        )
                    )

            fig_local.add_trace(
                go.Scatter(
                    x=x_vals_out,
                    y=y_sample_out.values,
                    mode='lines',
                    line={'color': 'rgba(239, 85, 59, 1.0)', 'width': 2},
                    name=sample_id
                )
            )

            fig_local.update_layout(
                template='plotly_white',
                margin=dict(l=40, r=20, t=40, b=40),
                height=320,
                legend=dict(orientation='h'),
                title=title
            )

            if x_is_categorical:
                fig_local.update_xaxes(categoryorder='array', categoryarray=x_vals_out)
            return fig_local

        children = []

        if core_col_names:
            # Load core data for this sample
            df_sample_core = db.lazy_load_core_samples(fresh_client, sample_ids=[sample_id], features=core_col_names)
            if not df_sample_core.empty:
                y_sample_core = pd.to_numeric(df_sample_core.iloc[0], errors='coerce')
                if not df_core_stats.empty:
                    mean_core = df_core_stats.reindex(y_sample_core.index).get('mean')
                    std_core = df_core_stats.reindex(y_sample_core.index).get('std').fillna(0.0)
                else:
                    mean_core = None
                    std_core = None
                fig_core = make_feature_comparison_figure('Fragment size distribution (core)', core_x, y_sample_core, mean_core, std_core)
                fig_core.update_xaxes(title_text='Core position')
                fig_core.update_yaxes(title_text='Value')
                children.append(dcc.Graph(figure=fig_core, config={'responsive': True}))

        if csm_col_names:
            # Load CSM data for this sample
            df_sample_csm = db.lazy_load_core_samples(fresh_client, sample_ids=[sample_id], features=csm_col_names)
            if not df_sample_csm.empty:
                y_sample_csm = pd.to_numeric(df_sample_csm.iloc[0], errors='coerce')
                if not df_core_stats.empty:
                    mean_csm = df_core_stats.reindex(y_sample_csm.index).get('mean')
                    std_csm = df_core_stats.reindex(y_sample_csm.index).get('std').fillna(0.0)
                else:
                    mean_csm = None
                    std_csm = None
                fig_csm = make_feature_comparison_figure('CSM features', csm_x, y_sample_csm, mean_csm, std_csm, sort_x_by_deviation=sort_motifs, x_is_categorical=True)
                fig_csm.update_xaxes(title_text='Motif')
                fig_csm.update_yaxes(title_text='Value')
                children.append(dcc.Graph(figure=fig_csm, config={'responsive': True}))

        if p5_col_names:
            # Load 5p data for this sample
            df_sample_p5 = db.lazy_load_core_samples(fresh_client, sample_ids=[sample_id], features=p5_col_names)
            if not df_sample_p5.empty:
                y_sample_p5 = pd.to_numeric(df_sample_p5.iloc[0], errors='coerce')
                if not df_core_stats.empty:
                    mean_p5 = df_core_stats.reindex(y_sample_p5.index).get('mean')
                    std_p5 = df_core_stats.reindex(y_sample_p5.index).get('std').fillna(0.0)
                else:
                    mean_p5 = None
                    std_p5 = None
                fig_p5 = make_feature_comparison_figure("5' features", p5_x, y_sample_p5, mean_p5, std_p5, sort_x_by_deviation=sort_motifs, x_is_categorical=True)
                fig_p5.update_xaxes(title_text='Motif')
                fig_p5.update_yaxes(title_text='Value')
                children.append(dcc.Graph(figure=fig_p5, config={'responsive': True}))

        if not children:
            return "No core/CSM/5p features available to plot for this dataset."

        return html.Div(children)

    app.run(debug=False, host="0.0.0.0", port=8050)