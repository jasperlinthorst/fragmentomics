import numpy as np
import pandas as pd
import pickle
import copy
import time
import warnings
import threading
import os
import uuid
import logging
import base64
import io
import gzip
import tempfile
import html as html_mod

import dash
from dash import dcc, html
from dash import dash_table
from dash.dependencies import Input, Output, State, ALL

import plotly.express as px
import plotly.graph_objects as go
from flask import request, jsonify

from cfstats import fszd, csm, fpends, db

warnings.filterwarnings(
    'ignore',
    message=r'.*The copy keyword is deprecated and will be removed in a future version\..*',
    category=FutureWarning,
)


def _decode_upload_contents_to_bytes(contents):
    if contents is None:
        return b''
    b64data = contents.split(',', 1)[1] if isinstance(contents, str) and ',' in contents else contents
    if isinstance(b64data, bytes):
        raw = b64data
    else:
        s = str(b64data)
        s = ''.join(s.split())
        if s == '':
            return b''
        pad = (-len(s)) % 4
        if pad:
            s += '=' * pad
        try:
            raw = base64.b64decode(s, validate=True)
        except Exception:
            raw = base64.b64decode(s)

    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    return raw


def explore(args):

    log = logging.getLogger(__name__)

    t0 = time.time()
    log.info('Connecting to ClickHouse')

    ch_client = db.get_client(
        host=getattr(args, 'ch_host', 'localhost'),
        port=getattr(args, 'ch_port', 8123)
    )

    # Check if we have data without loading full tables
    has_data = db.get_sample_count(ch_client) > 0
    
    log.info('Data check complete in %.2fs (has_data=%s)', time.time() - t0, has_data)

    # Get feature names lazily for dropdowns (meta-only = shown in UI)
    meta_features = db.get_meta_features(ch_client, feature_type='meta') if has_data else []
    numeric_columns = meta_features
    color_columns = meta_features
    
    # Get feature groups (fszd, csm, 5p, bincount) from meta_features by pattern
    _empty_fg = {
        'fszd': {'col_names': [], 'x_vals': []},
        'csm': {'col_names': [], 'x_vals': []},
        '5p': {'col_names': [], 'x_vals': []},
        'bincount': {'col_names': [], 'chromosomes': [], 'starts': [], 'ends': []},
    }
    feature_groups = db.get_feature_groups(ch_client) if has_data else _empty_fg

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

    upload_max_size = 3 * 1024 * 1024 * 1024
    app.server.config['MAX_CONTENT_LENGTH'] = upload_max_size

    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='selected-sample', data=None),
        dcc.Store(id='uploaded-point', data=[]),
        dcc.Store(id='admin-auth', data=False),
        dcc.Store(id='filter-indices', data=[]),
        dcc.Store(id='filters-store', data=[]),
        dcc.Store(id='hist-quantiles', data=4),
        html.Div(id='page-content'),
    ])

    meta_series_cache = {}
    meta_series_lock = threading.Lock()

    def _get_meta_series(feature_name):
        feature_name = str(feature_name)
        with meta_series_lock:
            if feature_name in meta_series_cache:
                return meta_series_cache[feature_name]

        fresh_client = db.get_fresh_client(
            host=getattr(args, 'ch_host', 'localhost'),
            port=getattr(args, 'ch_port', 8123)
        )
        s = db.load_meta_feature_series(fresh_client, feature_name)

        with meta_series_lock:
            meta_series_cache[feature_name] = s
        return s

    def _get_meta_wide(features):
        features = [str(f) for f in (features or []) if f is not None]
        features = list(dict.fromkeys(features))
        if not features:
            return pd.DataFrame()

        series_list = []
        for f in features:
            s = _get_meta_series(f)
            if hasattr(s, 'index') and getattr(s.index, 'has_duplicates', False):
                s = s[~s.index.duplicated(keep='last')]
            series_list.append(s)

        df_wide = pd.concat(series_list, axis=1)
        if getattr(df_wide.index, 'has_duplicates', False):
            df_wide = df_wide.loc[~df_wide.index.duplicated(keep='last')]
        df_wide.index = df_wide.index.astype(str)
        df_wide['_sample_id'] = df_wide.index.astype(str)
        return df_wide

    def _effective_detail_features(selected_columns, selected_x=None, selected_y=None, selected_color=None):
        if selected_columns and isinstance(selected_columns, (list, tuple)) and len(selected_columns) > 0:
            return list(selected_columns)
        features = []
        if selected_x:
            features.append(selected_x)
        if selected_y and selected_y not in features:
            features.append(selected_y)
        if selected_color and selected_color not in features:
            features.append(selected_color)
        return features

    def _filter_feature_names(filters):
        if not filters:
            return []
        out = []
        for f in filters:
            if isinstance(f, dict) and f.get('column'):
                out.append(str(f.get('column')))
        return list(dict.fromkeys(out))

    def _classify_color_series(series):
        if series is None:
            return {'mode': 'none', 'unique_count': 0, 'is_numeric': False, 'is_integer_like': False}

        s = series.dropna()
        if s.empty:
            return {'mode': 'none', 'unique_count': 0, 'is_numeric': False, 'is_integer_like': False}

        if not pd.api.types.is_numeric_dtype(s):
            return {'mode': 'categorical', 'unique_count': int(s.nunique()), 'is_numeric': False, 'is_integer_like': False}

        numeric = pd.to_numeric(s, errors='coerce').dropna()
        if numeric.empty:
            return {'mode': 'none', 'unique_count': 0, 'is_numeric': True, 'is_integer_like': False}

        values = numeric.to_numpy(dtype=float)
        is_integer_like = bool(np.all(np.isclose(values, np.round(values), rtol=1e-9, atol=1e-9, equal_nan=True)))
        unique_count = int(pd.Series(numeric).nunique())

        if is_integer_like and unique_count <= 10:
            mode = 'categorical'
        else:
            mode = 'continuous'

        return {
            'mode': mode,
            'unique_count': unique_count,
            'is_numeric': True,
            'is_integer_like': is_integer_like,
        }

    def _prepare_histogram_color_dataframe(df, selected_color, hist_quantiles):
        if selected_color is None or selected_color not in df.columns:
            return df, None, None, {'mode': 'none', 'unique_count': 0, 'is_numeric': False, 'is_integer_like': False}

        color_info = _classify_color_series(df[selected_color])
        if color_info['mode'] == 'none':
            return df, None, None, color_info

        if color_info['mode'] == 'categorical':
            return df, selected_color, selected_color, color_info

        df_plot = df[df[selected_color].notna()].copy()
        if df_plot.empty:
            return df_plot, None, None, {'mode': 'none', 'unique_count': 0, 'is_numeric': True, 'is_integer_like': color_info['is_integer_like']}

        quantiles = max(2, min(10, int(hist_quantiles or 4)))
        bin_col = f'__{selected_color}_quantile_bin'
        try:
            binned = pd.qcut(df_plot[selected_color], q=quantiles, duplicates='drop')
        except ValueError:
            return df_plot, selected_color, selected_color, {'mode': 'categorical', **{k: v for k, v in color_info.items() if k != 'mode'}}

        if getattr(binned, 'cat', None) is not None and len(binned.cat.categories) <= 1:
            return df_plot, selected_color, selected_color, {'mode': 'categorical', **{k: v for k, v in color_info.items() if k != 'mode'}}

        df_plot[bin_col] = binned.astype(str)
        return df_plot, bin_col, f'{selected_color} quantile', color_info

    def _hist_quantile_controls_visible(selected_x, selected_y, selected_color, df_filtered):
        if not selected_color:
            return False
        if (selected_x is None) == (selected_y is None):
            return False
        if df_filtered is None or df_filtered.empty or selected_color not in df_filtered.columns:
            return False
        color_info = _classify_color_series(df_filtered[selected_color])
        return color_info['mode'] == 'continuous'

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
                    value=None,
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

        html.Div([
            html.Span('Histogram color quantiles:', style={'marginRight': '10px'}),
            html.Button('-', id='hist-quantiles-minus', n_clicks=0),
            html.Span(id='hist-quantiles-label', style={'padding': '0 12px'}, children='4'),
            html.Button('+', id='hist-quantiles-plus', n_clicks=0),
        ], id='hist-quantiles-controls', style={'display': 'none', 'padding': '0 10px 10px 10px', 'alignItems': 'center', 'gap': '8px'}),

        dcc.Loading(
            id="loading-scatter",
            children=[dcc.Graph(id='scatter-plot')],
            style={'padding': '10px', 'width': '100%', 'height': '100%'},
            type="graph"
        ),

        dcc.Dropdown(
            id='detail-columns-dropdown',
            options=[{'label': col, 'value': col} for col in color_columns],
            value=[],
            multi=True,
            placeholder="Select columns to show in details"
        ),

        html.Div([
            dcc.Checklist(
                id='sort-motifs-by-deviation',
                options=[{'label': 'Sort motif plots by |sample - cloud mean|', 'value': 'sort'}],
                value=[],
                inline=True,
            ),
            html.Div([
                html.Label('Z-score threshold:', style={'marginRight': '6px'}),
                dcc.Input(
                    id='zscore-threshold',
                    type='number',
                    value=3,
                    min=0,
                    step=0.5,
                    style={'width': '70px'}
                ),
            ], style={'display': 'inline-flex', 'alignItems': 'center', 'marginLeft': '20px'}),
        ], style={'padding': '10px', 'display': 'flex', 'alignItems': 'center'}),

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

        dcc.Loading(
            id='loading-bincount-details',
            type='default',
            children=html.Div(
                id='point-details-bincount',
                style={'padding': '10px', 'width': '100%', 'overflowX': 'auto'}
            )
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
                    dcc.Input(id='admin-password-input', type='password', value='', n_submit=0, autoFocus=True, style={'margin': '0 10px'}),
                    html.Button('Login', id='admin-login-btn', n_clicks=0),
                    html.Div(id='admin-login-status', style={'padding': '10px'})
                ], style={'padding': '10px'})
            ])

        return html.Div([
            html.H3('Admin'),
            html.Div([
                dcc.Link('Back to explorer', href='/', style={'padding': '10px', 'display': 'inline-block'}),
            ]),
            html.Div(
                f"Database: {db.get_sample_count(ch_client)} samples",
                style={'padding': '5px 10px', 'color': '#888', 'fontSize': '12px'}
            ),
            html.Div([
                html.Button('Optimize meta table (FINAL)', id='admin-optimize-meta-btn', n_clicks=0),
                html.Div(id='admin-optimize-meta-status', style={'padding': '0px 10px 10px 10px'}),
            ], style={'padding': '10px'}),
            html.Div([
                html.A('Streaming uploads (large files)', href='/admin/stream-upload', target='_blank'),
            ], style={'padding': '10px'}),
            html.H4('Import TSV'),
            html.Div([
                html.Div([
                    html.Label('Feature type:', style={'fontWeight': 'bold', 'marginRight': '8px'}),
                    dcc.Dropdown(
                        id='admin-upload-feature-type',
                        options=[
                            {'label': 'Meta (shown in dropdowns)', 'value': 'meta'},
                            {'label': 'Core (hidden from dropdowns)', 'value': 'core'},
                        ],
                        value='meta',
                        clearable=False,
                        style={'width': '260px', 'display': 'inline-block', 'verticalAlign': 'middle'}
                    ),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px'}),
                html.Div([
                    html.Label('Description:', style={'fontWeight': 'bold', 'marginRight': '8px'}),
                    dcc.Input(
                        id='admin-upload-description',
                        type='text',
                        placeholder='How were these features calculated?',
                        style={'width': '400px'}
                    ),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px'}),
            ], style={'padding': '0 10px'}),
            dcc.Upload(
                id='admin-upload-meta-tsv',
                children=html.Div(['Drag and Drop or ', html.A('Select TSV')]),
                max_size=upload_max_size,
                style={
                    'width': '60%', 'height': '50px', 'lineHeight': '50px',
                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                    'textAlign': 'center', 'margin': '10px'
                },
                multiple=False
            ),
            dcc.Store(id='admin-tsv-preview-store', data=None),
            dcc.Store(id='admin-import-token-store', data=None),
            dcc.Interval(id='admin-import-progress-interval', interval=600,
                         disabled=True),
            html.Div(id='admin-tsv-columns-div', style={'padding': '0 10px'}),
            html.Div(id='admin-import-progress-div',
                     style={'padding': '0 10px', 'display': 'none'}),
            html.Div(id='admin-meta-status', style={'padding': '0px 10px 10px 10px'}),
            html.H4('Database Statistics'),
            html.Div([
                html.Button('Update Features & Stats', id='admin-update-features-btn', n_clicks=0,
                            style={'backgroundColor': '#17a2b8', 'color': 'white', 'border': 'none',
                                   'padding': '8px 16px', 'borderRadius': '4px', 'cursor': 'pointer', 'marginRight': '10px'}),
            ], style={'padding': '10px'}),
            html.Div(id='admin-update-stats-status', style={'padding': '0px 10px 10px 10px'}),
            html.H4('Drop Features'),
            html.Div([
                dcc.Dropdown(
                    id='admin-drop-features-dropdown',
                    options=[{'label': f, 'value': f} for f in db.get_meta_features(ch_client, feature_type=None)],
                    value=[],
                    multi=True,
                    placeholder='Select features to drop...',
                    style={'width': '80%', 'marginBottom': '8px'}
                ),
                html.Button('Drop selected features', id='admin-drop-features-btn', n_clicks=0,
                            style={'backgroundColor': '#dc3545', 'color': 'white', 'border': 'none',
                                   'padding': '8px 16px', 'borderRadius': '4px', 'cursor': 'pointer'}),
                html.Div(id='admin-drop-features-status', style={'padding': '6px 0', 'color': '#555'}),
            ], style={'padding': '10px'}),
            html.Div([
                html.Button('Clear database', id='admin-clear-db-btn', n_clicks=0,
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
        Output('hist-quantiles', 'data'),
        Input('hist-quantiles-minus', 'n_clicks'),
        Input('hist-quantiles-plus', 'n_clicks'),
        State('hist-quantiles', 'data'),
        prevent_initial_call=True
    )
    def update_hist_quantiles(minus_clicks, plus_clicks, current_quantiles):
        ctx = dash.callback_context
        current = max(2, min(10, int(current_quantiles or 4)))
        if not ctx.triggered:
            return current

        trig = ctx.triggered_id
        if trig == 'hist-quantiles-minus':
            return max(2, current - 1)
        if trig == 'hist-quantiles-plus':
            return min(10, current + 1)
        return current

    @app.callback(
        Output('hist-quantiles-controls', 'style'),
        Output('hist-quantiles-label', 'children'),
        Input('x-dropdown', 'value'),
        Input('y-dropdown', 'value'),
        Input('color-dropdown', 'value'),
        Input('filters-store', 'data'),
        Input('hist-quantiles', 'data')
    )
    def update_hist_quantile_controls(selected_x, selected_y, selected_color, filters, hist_quantiles):
        quantiles = max(2, min(10, int(hist_quantiles or 4)))

        if not _hist_quantile_controls_visible(selected_x, selected_y, selected_color, pd.DataFrame()):
            if not selected_color or (selected_x is None) == (selected_y is None):
                return {'display': 'none'}, str(quantiles)

        features_needed = []
        if selected_color:
            features_needed.append(selected_color)
        features_needed.extend(_filter_feature_names(filters))
        features_needed.append('_sample_id')

        df_wide = _get_meta_wide(features_needed)
        if df_wide.empty:
            return {'display': 'none'}, str(quantiles)

        mask = db.apply_filters_pandas(df_wide, filters)
        df_filtered = df_wide.loc[mask]

        if not _hist_quantile_controls_visible(selected_x, selected_y, selected_color, df_filtered):
            return {'display': 'none'}, str(quantiles)

        return {
            'display': 'flex',
            'padding': '0 10px 10px 10px',
            'alignItems': 'center',
            'gap': '8px'
        }, str(quantiles)

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
        Output('uploaded-point', 'data'),
        Output('upload-status', 'children'),
        Output('x-dropdown', 'value'),
        Output('y-dropdown', 'value'),
        Output('loading-status', 'children'),
        Input('upload-alignment', 'contents'),
        State('upload-alignment', 'filename'),
        State('x-dropdown', 'value'),
        State('y-dropdown', 'value'),
        State('uploaded-point', 'data')
    )
    def handle_alignment_upload(contents, filename, current_x, current_y, existing_points):
        existing_points = existing_points or []
        if contents is None:
            log.info('User cancelled alignment upload')
            return existing_points, '', current_x, current_y, 'Ready'

        if not isinstance(contents, str):
            return existing_points, f'Upload failed: unexpected contents type {type(contents)}.', current_x, current_y, 'Error: Invalid content type'

        # Show processing status before validation
        processing_status = '🔄 Processing uploaded file...'

        if mapping is None:
            return existing_points, 'Upload disabled: start the app with --mapping <mapping.pkl> so I can compute x/y coordinates.', current_x, current_y, 'Error: Mapping not loaded'

        if args.reference is None:
            return existing_points, 'Upload disabled: provide --reference <fasta> (required for motif features).', current_x, current_y, 'Error: Reference not provided'

        try:
            header, b64data = contents.split(',', 1)
            data = base64.b64decode(b64data)
        except Exception:
            return existing_points, 'Upload failed: could not decode file contents.', current_x, current_y, processing_status

        ext = ''
        if filename:
            ext = os.path.splitext(filename)[1]
        if ext.lower() not in ['.bam', '.sam', '.cram']:
            return existing_points, f'Upload failed: unsupported file type {ext}. Please upload .bam, .sam, or .cram.', current_x, current_y, processing_status

        log.info('Processing alignment upload: %s', filename)

        try:
            b64data = contents.split(',', 1)[1] if ',' in contents else contents
            b64data = b64data.strip().replace('\n', '').replace('\r', '')
            pad = (-len(b64data)) % 4
            if pad:
                b64data += '=' * pad

            data = base64.b64decode(b64data)
        except Exception as e:
            return existing_points, f'Upload failed: could not decode file contents ({type(e).__name__}: {str(e)}).', current_x, current_y, processing_status

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
            
            # Build feature names from current feature groups
            fg = db.get_feature_groups(ch_client)
            all_feature_names = fg['fszd']['col_names'] + fg['csm']['col_names'] + fg['5p']['col_names']
            
            # Insert results into ClickHouse
            try:
                sample_id = db.insert_alignment_results(ch_client, filename, f, all_feature_names, x_new, y_new)
                if sample_id:
                    log.info('Inserted alignment results into ClickHouse with sample_id: %s', sample_id)
                else:
                    log.warning('Failed to insert alignment results into ClickHouse')
            except Exception as e:
                log.exception('Failed to insert alignment results into ClickHouse')
            
            log.info('Alignment processed: %s (umap1=%.3f, umap2=%.3f)', label, x_new, y_new)
            new_point = {
                'umap1': x_new,
                'umap2': y_new,
                'label': label,
            }
            return existing_points + [new_point], f'Processed: {label} (umap1={x_new:.3f}, umap2={y_new:.3f})', 'umap1', 'umap2', f'✓ Processed: {label}'
        except Exception as e:
            log.exception('Alignment upload failed for %s', filename)
            return existing_points, f'Upload failed: {str(e)}', current_x, current_y, f'Error: {str(e)}'
        finally:
            pass

    @app.callback(
        Output('admin-auth', 'data'),
        Output('admin-login-status', 'children'),
        Input('admin-login-btn', 'n_clicks'),
        Input('admin-password-input', 'n_submit'),
        State('admin-password-input', 'value'),
        prevent_initial_call=True
    )
    def admin_login(n_clicks, n_submit, password_input):
        if not n_clicks and not n_submit:
            return dash.no_update, ''
        if admin_password is None:
            return False, 'Admin page disabled.'
        if password_input == admin_password:
            log.info('Admin login succeeded')
            return True, '✓ Logged in.'
        log.warning('Admin login failed')
        return False, '✗ Invalid password.'

    _DTYPE_OPTIONS = [
        {'label': t, 'value': t} for t in db.SUPPORTED_DATA_TYPES
    ]

    # ── Step 1: Analyze uploaded TSV and show per-column type selectors ──
    @app.callback(
        Output('admin-tsv-preview-store', 'data'),
        Output('admin-tsv-columns-div', 'children'),
        Input('admin-upload-meta-tsv', 'contents'),
        State('admin-upload-meta-tsv', 'filename'),
        prevent_initial_call=True
    )
    def admin_analyze_tsv(contents, filename):
        if contents is None:
            return None, ''
        try:
            raw = _decode_upload_contents_to_bytes(contents)
            columns, id_col = db.analyze_tsv_columns(ch_client, raw)
            log.info('Analyzed TSV %s: %d feature columns, id_col=%s', filename, len(columns), id_col)

            rows = []
            for col in columns:
                is_existing = col['existing_type'] is not None
                chosen_type = col['existing_type'] if is_existing else col['detected_type']

                vals = col.get('sample_values', [])
                has_more = col.get('has_more', False)
                vals_str = ', '.join(str(v) for v in vals)
                if has_more:
                    vals_str += ', ...'

                rows.append(html.Tr([
                    html.Td(
                        dcc.Checklist(
                            id={'type': 'col-include', 'index': col['name']},
                            options=[{'label': '', 'value': 'on'}],
                            value=['on'],
                            style={'display': 'inline-block'}
                        ), style={'padding': '4px 4px', 'width': '30px', 'textAlign': 'center'}
                    ),
                    html.Td(
                        dcc.Input(
                            id={'type': 'col-rename', 'index': col['name']},
                            value=col['name'],
                            type='text',
                            style={'width': '200px', 'fontFamily': 'monospace',
                                   'fontSize': '12px', 'padding': '2px 4px',
                                   'border': '1px solid #ccc', 'borderRadius': '3px'},
                        ), style={'padding': '4px 4px'}
                    ),
                    html.Td(
                        dcc.Dropdown(
                            id={'type': 'col-dtype', 'index': col['name']},
                            options=_DTYPE_OPTIONS,
                            value=chosen_type,
                            disabled=is_existing,
                            clearable=False,
                            style={'width': '120px',
                                   'opacity': '0.5' if is_existing else '1'}
                        ), style={'padding': '4px 4px'}
                    ),
                    html.Td(
                        f"Existing ({col['existing_type']})" if is_existing else "Auto-detected",
                        style={'padding': '4px 8px', 'color': '#888' if is_existing else '#28a745',
                               'fontSize': '12px', 'whiteSpace': 'nowrap'}
                    ),
                    html.Td(vals_str,
                             style={'padding': '4px 8px', 'fontSize': '11px', 'color': '#666',
                                    'maxWidth': '350px', 'overflow': 'hidden',
                                    'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'}),
                ]))

            ui = html.Div([
                html.P(f'File: {filename} — {len(columns)} feature columns (id column: {id_col})',
                       style={'fontWeight': 'bold', 'marginBottom': '6px'}),
                html.Div([
                    html.Label('Set all types:', style={'marginRight': '6px', 'fontSize': '13px'}),
                    dcc.Dropdown(
                        id='admin-set-all-dtype',
                        options=_DTYPE_OPTIONS,
                        placeholder='Choose...',
                        clearable=True,
                        style={'width': '130px', 'display': 'inline-block', 'verticalAlign': 'middle'},
                    ),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px'}),
                html.Div(
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th('', style={'padding': '4px 4px', 'width': '30px'}),
                            html.Th('Name (editable)', style={'padding': '4px 4px', 'textAlign': 'left'}),
                            html.Th('Data type', style={'padding': '4px 4px', 'textAlign': 'left'}),
                            html.Th('Status', style={'padding': '4px 8px', 'textAlign': 'left'}),
                            html.Th('Sample values', style={'padding': '4px 8px', 'textAlign': 'left'}),
                        ])),
                        html.Tbody(rows),
                    ], style={'borderCollapse': 'collapse', 'fontSize': '13px', 'width': '100%'}),
                    style={'maxHeight': '400px', 'overflowY': 'auto', 'border': '1px solid #ddd',
                           'borderRadius': '4px', 'marginBottom': '10px'}
                ),
                html.Button('Confirm & Import', id='admin-tsv-confirm-btn', n_clicks=0,
                            style={'backgroundColor': '#28a745', 'color': 'white', 'border': 'none',
                                   'padding': '8px 20px', 'borderRadius': '4px', 'cursor': 'pointer',
                                   'fontSize': '14px'}),
            ])

            store_data = {
                'content_b64': base64.b64encode(raw).decode('ascii'),
                'filename': filename,
                'column_names': [c['name'] for c in columns],
            }
            return store_data, ui
        except Exception as e:
            log.exception('TSV analysis failed: %s', filename)
            return None, html.Div(f'✗ Analysis failed: {str(e)}', style={'color': 'red'})

    # ── Clientside callback: "Set all types" applies to non-disabled dropdowns ──
    app.clientside_callback(
        """
        function(setAll, currentVals, disabledStates) {
            if (!setAll) return window.dash_clientside.no_update;
            return currentVals.map(function(v, i) {
                return disabledStates[i] ? v : setAll;
            });
        }
        """,
        Output({'type': 'col-dtype', 'index': ALL}, 'value'),
        Input('admin-set-all-dtype', 'value'),
        State({'type': 'col-dtype', 'index': ALL}, 'value'),
        State({'type': 'col-dtype', 'index': ALL}, 'disabled'),
        prevent_initial_call=True,
    )

    # ── Step 2: Confirm import — start background thread, enable progress ──
    @app.callback(
        Output('admin-import-token-store', 'data'),
        Output('admin-import-progress-interval', 'disabled'),
        Output('admin-import-progress-div', 'style'),
        Output('admin-import-progress-div', 'children'),
        Output('admin-meta-status', 'children', allow_duplicate=True),
        Input('admin-tsv-confirm-btn', 'n_clicks'),
        State({'type': 'col-dtype', 'index': ALL}, 'value'),
        State({'type': 'col-dtype', 'index': ALL}, 'id'),
        State({'type': 'col-include', 'index': ALL}, 'value'),
        State({'type': 'col-include', 'index': ALL}, 'id'),
        State({'type': 'col-rename', 'index': ALL}, 'value'),
        State({'type': 'col-rename', 'index': ALL}, 'id'),
        State('admin-tsv-preview-store', 'data'),
        State('admin-upload-feature-type', 'value'),
        State('admin-upload-description', 'value'),
        prevent_initial_call=True
    )
    def admin_confirm_tsv_import(n_clicks, dtype_values, dtype_ids,
                                  include_values, include_ids,
                                  rename_values, rename_ids,
                                  preview_data, feat_type, feat_description):
        if not n_clicks or preview_data is None:
            raise dash.exceptions.PreventUpdate

        # Build per-column type dict (keyed by original name)
        col_types = {}
        for dd_id, dd_val in zip(dtype_ids, dtype_values):
            col_types[dd_id['index']] = dd_val

        # Build include list from checkboxes
        include_columns = []
        for cb_id, cb_val in zip(include_ids, include_values):
            if 'on' in (cb_val or []):
                include_columns.append(cb_id['index'])

        # Build rename map (original_name -> new_name), skip unchanged
        rename_columns = {}
        for rn_id, rn_val in zip(rename_ids, rename_values):
            orig = rn_id['index']
            new_name = (rn_val or '').strip()
            if new_name and new_name != orig:
                rename_columns[orig] = new_name

        # Validate renames against existing DB features
        if rename_columns:
            new_names = set(rename_columns.values())
            try:
                existing_types = db._get_feature_data_types_bulk(ch_client, list(new_names))
            except Exception:
                existing_types = {}
            for orig, new_name in rename_columns.items():
                if new_name in existing_types:
                    expected_type = existing_types[new_name]
                    chosen_type = col_types.get(orig, 'String')
                    if chosen_type != expected_type:
                        err = (f'✗ Cannot rename "{orig}" to "{new_name}": '
                               f'feature already exists with type {expected_type}, '
                               f'but you selected {chosen_type}.')
                        return None, True, {'display': 'none'}, '', err

        raw = base64.b64decode(preview_data['content_b64'])
        filename = preview_data.get('filename', '?')
        token = uuid.uuid4().hex
        log.info('Admin confirmed TSV import: %s (%d/%d columns, %d renames, token=%s)',
                 filename, len(include_columns), len(col_types), len(rename_columns), token)

        def _do_import():
            try:
                import_client = db.get_fresh_client(
                    host=getattr(args, 'ch_host', 'localhost'),
                    port=getattr(args, 'ch_port', 8123))
                db.upload_tsv_from_bytes(
                    import_client, raw,
                    feature_type=feat_type or None,
                    description=feat_description or None,
                    col_types=col_types,
                    include_columns=include_columns,
                    rename_columns=rename_columns if rename_columns else None,
                    progress_token=token)
            except Exception as exc:
                log.exception('Background TSV import failed')
                db._set_upload_progress(token, 0, 0, 0, f'error: {exc}')

        threading.Thread(target=_do_import, daemon=True).start()

        progress_bar = html.Div([
            html.Div(style={
                'width': '0%', 'height': '18px', 'backgroundColor': '#28a745',
                'borderRadius': '4px', 'transition': 'width 0.3s',
            }, id='admin-progress-bar-inner'),
        ], style={
            'width': '100%', 'backgroundColor': '#e9ecef', 'borderRadius': '4px',
            'overflow': 'hidden', 'marginBottom': '6px',
        })
        progress_text = html.Div('Starting import...', id='admin-progress-text',
                                  style={'fontSize': '13px', 'color': '#555'})

        return (token, False,  # enable interval
                {'padding': '0 10px'},  # show progress div
                [progress_bar, progress_text],
                '')  # clear old status

    # ── Progress polling callback ──
    @app.callback(
        Output('admin-import-progress-div', 'children', allow_duplicate=True),
        Output('admin-import-progress-div', 'style', allow_duplicate=True),
        Output('admin-import-progress-interval', 'disabled', allow_duplicate=True),
        Output('admin-meta-status', 'children'),
        Input('admin-import-progress-interval', 'n_intervals'),
        State('admin-import-token-store', 'data'),
        prevent_initial_call=True
    )
    def admin_poll_import_progress(n_intervals, token):
        if not token:
            raise dash.exceptions.PreventUpdate
        prog = db.get_upload_progress(token)
        status = prog['status']
        current = prog['current']
        total = prog['total']
        rows_ins = prog['rows_inserted']

        pct = int(current / total * 100) if total > 0 else 0
        pct = min(pct, 100)

        bar_inner = html.Div(style={
            'width': f'{pct}%', 'height': '18px', 'backgroundColor': '#28a745',
            'borderRadius': '4px', 'transition': 'width 0.3s',
        })
        bar_outer = html.Div([bar_inner], style={
            'width': '100%', 'backgroundColor': '#e9ecef', 'borderRadius': '4px',
            'overflow': 'hidden', 'marginBottom': '6px',
        })

        if status == 'done':
            db.clear_upload_progress(token)
            # Refresh feature lists
            nonlocal color_columns, numeric_columns, feature_groups
            meta_features = db.get_meta_features(ch_client, feature_type='meta')
            color_columns = meta_features
            numeric_columns = meta_features
            feature_groups = db.get_feature_groups(ch_client)

            return ([], {'display': 'none'}, True,
                    f'✓ Imported {rows_ins} value rows. Features updated.')
        elif status.startswith('error'):
            db.clear_upload_progress(token)
            return ([], {'display': 'none'}, True,
                    f'✗ Import failed: {status}')
        elif status == 'updating_features':
            text = html.Div(f'{pct}% — Updating feature statistics...',
                             style={'fontSize': '13px', 'color': '#555'})
            return ([bar_outer, text], {'padding': '0 10px'}, False, '')
        else:
            text = html.Div(f'{pct}% — {current:,}/{total:,} rows read, '
                             f'{rows_ins:,} values inserted',
                             style={'fontSize': '13px', 'color': '#555'})
            return ([bar_outer, text], {'padding': '0 10px'}, False, '')

    # Server-side pending streaming uploads: token -> temp file path
    _pending_stream_uploads = {}

    @app.server.route('/admin/stream-upload', methods=['GET'])
    def stream_upload_page():
        mapping_available = mapping is not None and getattr(args, 'reference', None) is not None
        alignment_section = (
            '<h4>Import BAM / CRAM / SAM</h4>'
            '<form method="POST" action="/admin/stream-upload-alignment" enctype="multipart/form-data">'
            '<label>Sample ID (optional):</label> '
            '<input type="text" name="sample_id" placeholder="Leave blank to auto-generate" '
            'style="width:300px;margin-left:6px;" /><br/><br/>'
            '<input type="file" name="file" accept=".bam,.sam,.cram" />'
            '<button type="submit" style="margin-left:10px;">Upload &amp; process</button>'
            '</form>'
        ) if mapping_available else (
            '<h4>Import BAM / CRAM / SAM</h4>'
            '<p style="color:#888;">Alignment upload disabled: start the app with '
            '<code>--mapping</code> and <code>--reference</code> flags.</p>'
        )
        return (
            '<h3>Streaming uploads</h3>'
            '<p>Use this page for very large files. This avoids Dash base64 uploads.</p>'
            '<h4>Import TSV</h4>'
            '<form method="POST" action="/admin/stream-upload-tsv" enctype="multipart/form-data">'
            '<label>Feature type:</label> '
            '<select name="feature_type" style="margin-left:6px;">'
            '<option value="meta" selected>Meta (shown in dropdowns)</option>'
            '<option value="core">Core (hidden from dropdowns)</option>'
            '</select><br/><br/>'
            '<label>Description:</label> '
            '<input type="text" name="description" placeholder="How were these features calculated?" '
            'style="width:400px;margin-left:6px;" /><br/><br/>'
            '<input type="file" name="file" accept=".tsv,.txt,.gz" />'
            '<button type="submit" style="margin-left:10px;">Analyze &amp; preview</button>'
            '</form>'
            '<hr/>'
            + alignment_section +
            '<hr/>'
            '<h4>Import PLINK BED/BIM/FAM (genotypes)</h4>'
            '<p style="font-size:13px;color:#555;">Upload PLINK binary genotype files. '
            'Each variant becomes a UInt8 feature (0=hom ref, 1=het, 2=hom alt). '
            'Feature type is set to <b>core</b>.</p>'
            '<form method="POST" action="/admin/stream-upload-plink" enctype="multipart/form-data">'
            '<label>BED file:</label> '
            '<input type="file" name="bed" accept=".bed" style="margin-left:6px;" /><br/><br/>'
            '<label>BIM file:</label> '
            '<input type="file" name="bim" accept=".bim" style="margin-left:6px;" /><br/><br/>'
            '<label>FAM file:</label> '
            '<input type="file" name="fam" accept=".fam" style="margin-left:6px;" /><br/><br/>'
            '<button type="submit" style="margin-left:0;">Upload &amp; import</button>'
            '</form>'
            '<hr/>'
            '<p><a href="/admin">Back</a></p>'
        )

    @app.server.route('/admin/upload-progress/<token>')
    def upload_progress_json(token):
        """JSON endpoint for polling import progress."""
        return jsonify(db.get_upload_progress(token))

    @app.server.route('/admin/stream-upload-tsv', methods=['POST'])
    def stream_upload_tsv_analyze():
        """Step 1: save file, detect column types, show per-column selectors."""
        f = request.files.get('file')
        if f is None:
            return '<p>No file provided.</p><p><a href="/admin/stream-upload">Back</a></p>', 400

        feat_type = request.form.get('feature_type') or 'meta'
        feat_desc = request.form.get('description') or ''

        tmp = tempfile.NamedTemporaryFile(prefix='tsv_upload_', suffix='.tsv', delete=False)
        tmp_path = tmp.name
        tmp.close()
        f.save(tmp_path)

        try:
            columns, id_col = db.analyze_tsv_columns(ch_client, tmp_path)
        except Exception as e:
            log.exception('Streaming TSV analysis failed')
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return f'<p>Analysis failed: {str(e)}</p><p><a href="/admin/stream-upload">Back</a></p>', 500

        file_token = uuid.uuid4().hex
        _pending_stream_uploads[file_token] = tmp_path

        # Build "set all types" options
        set_all_options = ''.join(
            f'<option value="{dt}">{dt}</option>' for dt in db.SUPPORTED_DATA_TYPES
        )

        table_rows = []
        for col in columns:
            is_existing = col['existing_type'] is not None
            chosen = col['existing_type'] if is_existing else col['detected_type']
            esc_name = html_mod.escape(col['name'])
            options_html = ''.join(
                f'<option value="{dt}"{" selected" if dt == chosen else ""}>{dt}</option>'
                for dt in db.SUPPORTED_DATA_TYPES
            )
            disabled = ' disabled' if is_existing else ''
            opacity = 'opacity:0.5;' if is_existing else ''
            status = (f'<span style="color:#888;">Existing ({col["existing_type"]})</span>'
                      if is_existing else '<span style="color:#28a745;">Auto-detected</span>')
            hidden_input = (
                f'<input type="hidden" name="dtype_{esc_name}" value="{chosen}" />'
                if is_existing else ''
            )
            vals = col.get('sample_values', [])
            has_more = col.get('has_more', False)
            vals_str = ', '.join(html_mod.escape(str(v)) for v in vals)
            if has_more:
                vals_str += ', ...'

            table_rows.append(
                f'<tr>'
                f'<td style="padding:4px 4px;text-align:center;width:30px;">'
                f'<input type="checkbox" name="include_{esc_name}" value="on" checked />'
                f'</td>'
                f'<td style="padding:4px 4px;">'
                f'<input type="text" name="rename_{esc_name}" value="{esc_name}" '
                f'style="width:200px;font-family:monospace;font-size:12px;padding:2px 4px;'
                f'border:1px solid #ccc;border-radius:3px;" />'
                f'</td>'
                f'<td style="padding:4px 4px;">'
                f'<select name="dtype_{esc_name}" class="dtype-select" '
                f'style="width:120px;{opacity}"{disabled}>{options_html}</select>'
                f'{hidden_input}'
                f'</td>'
                f'<td style="padding:4px 8px;font-size:12px;">{status}</td>'
                f'<td style="padding:4px 8px;font-size:11px;color:#666;max-width:350px;'
                f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{vals_str}</td>'
                f'</tr>'
            )

        set_all_js = (
            '<script>'
            'document.getElementById("set-all-dtype").addEventListener("change", function(){'
            '  var v = this.value; if(!v) return;'
            '  document.querySelectorAll("select.dtype-select").forEach(function(s){'
            '    if(!s.disabled) s.value = v;'
            '  });'
            '});'
            '</script>'
        )

        return (
            '<h3>Review column types</h3>'
            f'<p><b>File:</b> {html_mod.escape(f.filename)} &mdash; {len(columns)} feature columns '
            f'(id column: <code>{html_mod.escape(id_col)}</code>)</p>'
            '<div style="margin-bottom:8px;">'
            '<label style="font-size:13px;margin-right:6px;">Set all types:</label>'
            f'<select id="set-all-dtype" style="width:130px;"><option value="">Choose...</option>'
            f'{set_all_options}</select></div>'
            '<form method="POST" action="/admin/stream-upload-tsv-confirm">'
            f'<input type="hidden" name="file_token" value="{file_token}" />'
            f'<input type="hidden" name="feature_type" value="{html_mod.escape(feat_type)}" />'
            f'<input type="hidden" name="description" value="{html_mod.escape(feat_desc)}" />'
            '<div style="max-height:400px;overflow-y:auto;border:1px solid #ddd;'
            'border-radius:4px;margin-bottom:12px;">'
            '<table style="border-collapse:collapse;font-size:13px;width:100%;">'
            '<thead><tr>'
            '<th style="padding:4px 4px;width:30px;"></th>'
            '<th style="padding:4px 4px;text-align:left;">Name (editable)</th>'
            '<th style="padding:4px 4px;text-align:left;">Data type</th>'
            '<th style="padding:4px 8px;text-align:left;">Status</th>'
            '<th style="padding:4px 8px;text-align:left;">Sample values</th>'
            '</tr></thead><tbody>'
            + ''.join(table_rows) +
            '</tbody></table></div>'
            '<button type="submit" style="background-color:#28a745;color:white;border:none;'
            'padding:8px 20px;border-radius:4px;cursor:pointer;font-size:14px;">'
            'Confirm &amp; Import</button>'
            '</form>'
            '<p><a href="/admin/stream-upload">Cancel</a></p>'
            + set_all_js
        )

    @app.server.route('/admin/stream-upload-tsv-confirm', methods=['POST'])
    def stream_upload_tsv_confirm():
        """Step 2: start background import, show progress page."""
        file_token = request.form.get('file_token')
        if not file_token or file_token not in _pending_stream_uploads:
            return '<p>Invalid or expired upload token.</p><p><a href="/admin/stream-upload">Back</a></p>', 400

        tmp_path = _pending_stream_uploads.pop(file_token)
        feat_type = request.form.get('feature_type') or None
        feat_desc = request.form.get('description') or None

        # Collect per-column types, includes, and renames from form fields
        col_types = {}
        include_columns = []
        rename_columns = {}
        for key, val in request.form.items():
            if key.startswith('dtype_'):
                col_name = key[len('dtype_'):]
                col_types[col_name] = val
            elif key.startswith('include_'):
                col_name = key[len('include_'):]
                if val == 'on':
                    include_columns.append(col_name)
            elif key.startswith('rename_'):
                orig = key[len('rename_'):]
                new_name = val.strip()
                if new_name and new_name != orig:
                    rename_columns[orig] = new_name

        # Validate renames against existing DB features
        if rename_columns:
            new_names = set(rename_columns.values())
            try:
                existing_types = db._get_feature_data_types_bulk(ch_client, list(new_names))
            except Exception:
                existing_types = {}
            for orig, new_name in rename_columns.items():
                if new_name in existing_types:
                    expected = existing_types[new_name]
                    chosen = col_types.get(orig, 'String')
                    if chosen != expected:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                        return (
                            f'<p style="color:red;">Cannot rename "{html_mod.escape(orig)}" '
                            f'to "{html_mod.escape(new_name)}": feature already exists with '
                            f'type {expected}, but you selected {chosen}.</p>'
                            '<p><a href="/admin/stream-upload">Back</a></p>'
                        ), 400

        progress_token = uuid.uuid4().hex

        def _do_stream_import():
            try:
                import_client = db.get_fresh_client(
                    host=getattr(args, 'ch_host', 'localhost'),
                    port=getattr(args, 'ch_port', 8123))
                db.upload_meta_tsv(
                    import_client, tmp_path,
                    feature_type=feat_type, description=feat_desc,
                    col_types=col_types,
                    include_columns=include_columns if include_columns else None,
                    rename_columns=rename_columns if rename_columns else None,
                    progress_token=progress_token)
            except Exception as exc:
                log.exception('Streaming background TSV import failed')
                db._set_upload_progress(progress_token, 0, 0, 0, f'error: {exc}')
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        threading.Thread(target=_do_stream_import, daemon=True).start()

        # Return progress page with JS polling
        return (
            '<h3>Importing...</h3>'
            '<div style="width:100%;background-color:#e9ecef;border-radius:4px;'
            'overflow:hidden;margin-bottom:8px;">'
            '<div id="prog-bar" style="width:0%;height:20px;background-color:#28a745;'
            'border-radius:4px;transition:width 0.3s;"></div></div>'
            '<p id="prog-text" style="font-size:13px;color:#555;">Starting import...</p>'
            '<div id="prog-done" style="display:none;">'
            '<p><a href="/admin">Back to admin</a> &middot; '
            '<a href="/admin/stream-upload">Upload another</a></p></div>'
            '<script>'
            f'var token = "{progress_token}";'
            'function poll(){'
            '  fetch("/admin/upload-progress/" + token).then(r=>r.json()).then(function(d){'
            '    var pct = d.total > 0 ? Math.min(Math.round(d.current/d.total*100),100) : 0;'
            '    document.getElementById("prog-bar").style.width = pct + "%";'
            '    if(d.status === "done"){'
            '      document.getElementById("prog-text").innerHTML = '
            '        "&#10003; Done! " + d.rows_inserted.toLocaleString() + " value rows imported.";'
            '      document.getElementById("prog-done").style.display = "block";'
            '    } else if(d.status.startsWith("error")){'
            '      document.getElementById("prog-text").innerHTML = '
            '        "<span style=\\"color:red;\\">Import failed: " + d.status + "</span>";'
            '      document.getElementById("prog-done").style.display = "block";'
            '    } else if(d.status === "updating_features"){'
            '      document.getElementById("prog-text").textContent = '
            '        pct + "% — Updating feature statistics...";'
            '      setTimeout(poll, 800);'
            '    } else {'
            '      document.getElementById("prog-text").textContent = '
            '        pct + "% — " + d.current.toLocaleString() + "/" + d.total.toLocaleString() '
            '        + " rows read, " + d.rows_inserted.toLocaleString() + " values inserted";'
            '      setTimeout(poll, 600);'
            '    }'
            '  }).catch(function(){ setTimeout(poll, 1000); });'
            '}'
            'setTimeout(poll, 500);'
            '</script>'
        )

    @app.server.route('/admin/stream-upload-alignment', methods=['POST'])
    def stream_upload_alignment():
        if mapping is None or getattr(args, 'reference', None) is None:
            return '<p>Alignment upload disabled: --mapping and --reference required.</p><p><a href="/admin/stream-upload">Back</a></p>', 400

        f = request.files.get('file')
        if f is None:
            return '<p>No file provided.</p><p><a href="/admin/stream-upload">Back</a></p>', 400

        filename = f.filename or 'upload'
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ('.bam', '.sam', '.cram'):
            return f'<p>Unsupported file type: {ext}. Use .bam, .sam, or .cram.</p><p><a href="/admin/stream-upload">Back</a></p>', 400

        custom_sample_id = (request.form.get('sample_id') or '').strip() or None

        uploads_dir = os.path.join(os.getcwd(), 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(prefix='align_upload_', suffix=ext, dir=uploads_dir, delete=False)
        tmp_path = tmp.name
        tmp.close()
        f.save(tmp_path)

        try:
            log.info('Streaming alignment upload: %s -> %s', filename, tmp_path)

            upload_args = copy.copy(args)
            upload_args.samfiles = [tmp_path]
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
            Xcsm = np.array(csm.cleavesitemotifs(upload_args, cmdline=False))
            Xsem = np.array(fpends._5pends(upload_args, cmdline=False))
            feat = np.concatenate((Xfszd, Xcsm, Xsem), axis=1)

            fp = reducer.transform(feat)
            x_new = float(fp[0, 0])
            y_new = float(fp[0, 1])

            fg = db.get_feature_groups(ch_client)
            all_feature_names = fg['fszd']['col_names'] + fg['csm']['col_names'] + fg['5p']['col_names']

            sample_id = db.insert_alignment_results(
                ch_client, custom_sample_id or filename, feat, all_feature_names, x_new, y_new
            )

            log.info('Streaming alignment processed: %s (sample_id=%s, umap1=%.3f, umap2=%.3f)', filename, sample_id, x_new, y_new)
            return (
                f'<p>Processed: {filename}</p>'
                f'<p>Sample ID: {sample_id}</p>'
                f'<p>umap1={x_new:.3f}, umap2={y_new:.3f}</p>'
                f'<p><a href="/admin/stream-upload">Upload another</a> | <a href="/admin">Back to admin</a></p>'
            )
        except Exception as e:
            log.exception('Streaming alignment upload failed: %s', filename)
            return f'<p>Processing failed: {str(e)}</p><p><a href="/admin/stream-upload">Back</a></p>', 500

    @app.server.route('/admin/stream-upload-plink', methods=['POST'])
    def stream_upload_plink():
        """Upload PLINK BED/BIM/FAM genotype files and import into meta_uint8."""
        bed_file = request.files.get('bed')
        bim_file = request.files.get('bim')
        fam_file = request.files.get('fam')

        missing = []
        if bed_file is None or bed_file.filename == '':
            missing.append('BED')
        if bim_file is None or bim_file.filename == '':
            missing.append('BIM')
        if fam_file is None or fam_file.filename == '':
            missing.append('FAM')
        if missing:
            return (f'<p>Missing file(s): {", ".join(missing)}.</p>'
                    '<p><a href="/admin/stream-upload">Back</a></p>'), 400

        uploads_dir = os.path.join(os.getcwd(), 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)

        bed_tmp = tempfile.NamedTemporaryFile(prefix='plink_', suffix='.bed', dir=uploads_dir, delete=False)
        bim_tmp = tempfile.NamedTemporaryFile(prefix='plink_', suffix='.bim', dir=uploads_dir, delete=False)
        fam_tmp = tempfile.NamedTemporaryFile(prefix='plink_', suffix='.fam', dir=uploads_dir, delete=False)
        bed_path, bim_path, fam_path = bed_tmp.name, bim_tmp.name, fam_tmp.name
        bed_tmp.close(); bim_tmp.close(); fam_tmp.close()

        bed_file.save(bed_path)
        bim_file.save(bim_path)
        fam_file.save(fam_path)

        progress_token = uuid.uuid4().hex
        log.info('PLINK streaming upload: bed=%s bim=%s fam=%s token=%s',
                 bed_file.filename, bim_file.filename, fam_file.filename, progress_token)

        def _do_plink_import():
            try:
                import_client = db.get_fresh_client(
                    host=getattr(args, 'ch_host', 'localhost'),
                    port=getattr(args, 'ch_port', 8123))
                db.upload_plink_bed(import_client, bed_path, bim_path, fam_path,
                                    progress_token=progress_token)
            except Exception as exc:
                log.exception('Background PLINK import failed')
                db._set_upload_progress(progress_token, 0, 0, 0, f'error: {exc}')
            finally:
                for p in (bed_path, bim_path, fam_path):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

        threading.Thread(target=_do_plink_import, daemon=True).start()

        return (
            '<h3>Importing PLINK genotypes...</h3>'
            '<div style="width:100%;background-color:#e9ecef;border-radius:4px;'
            'overflow:hidden;margin-bottom:8px;">'
            '<div id="prog-bar" style="width:0%;height:20px;background-color:#28a745;'
            'border-radius:4px;transition:width 0.3s;"></div></div>'
            '<p id="prog-text" style="font-size:13px;color:#555;">Starting import...</p>'
            '<div id="prog-done" style="display:none;">'
            '<p><a href="/admin">Back to admin</a> &middot; '
            '<a href="/admin/stream-upload">Upload another</a></p></div>'
            '<script>'
            f'var token = "{progress_token}";'
            'function poll(){'
            '  fetch("/admin/upload-progress/" + token).then(r=>r.json()).then(function(d){'
            '    var pct = d.total > 0 ? Math.min(Math.round(d.current/d.total*100),100) : 0;'
            '    document.getElementById("prog-bar").style.width = pct + "%";'
            '    if(d.status === "done"){'
            '      document.getElementById("prog-text").innerHTML = '
            '        "&#10003; Done! " + d.rows_inserted.toLocaleString() + " genotype values imported.";'
            '      document.getElementById("prog-done").style.display = "block";'
            '    } else if(d.status.startsWith("error")){'
            '      document.getElementById("prog-text").innerHTML = '
            '        "<span style=\\"color:red;\\">Import failed: " + d.status + "</span>";'
            '      document.getElementById("prog-done").style.display = "block";'
            '    } else if(d.status === "updating_features"){'
            '      document.getElementById("prog-text").textContent = '
            '        pct + "% — Updating feature statistics...";'
            '      setTimeout(poll, 800);'
            '    } else {'
            '      document.getElementById("prog-text").textContent = '
            '        pct + "% — " + d.current.toLocaleString() + "/" + d.total.toLocaleString() '
            '        + " variants processed, " + d.rows_inserted.toLocaleString() + " values inserted";'
            '      setTimeout(poll, 600);'
            '    }'
            '  }).catch(function(){ setTimeout(poll, 1000); });'
            '}'
            'setTimeout(poll, 500);'
            '</script>'
        )

    @app.callback(
        Output('admin-optimize-meta-status', 'children'),
        Input('admin-optimize-meta-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def admin_optimize_meta_table(n_clicks):
        if not n_clicks:
            return ''
        try:
            log.warning('Admin requested OPTIMIZE TABLE meta FINAL')
            ch_client.command(f'OPTIMIZE TABLE {db.DATABASE_NAME}.{db.META_TABLE} FINAL')
            return '✓ OPTIMIZE TABLE meta FINAL completed.'
        except Exception as e:
            log.exception('Admin OPTIMIZE TABLE meta FINAL failed')
            return f'✗ OPTIMIZE failed: {str(e)}'

    @app.callback(
        Output('admin-clear-db-status', 'children'),
        Input('admin-clear-db-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def admin_clear_db(n_clicks):
        nonlocal color_columns, numeric_columns, feature_groups
        if not n_clicks:
            return ''
        try:
            log.warning('Admin requested database clear')
            db.drop_tables(ch_client)
            color_columns = []
            numeric_columns = []
            feature_groups = _empty_fg
            return '✓ Database cleared.'
        except Exception as e:
            log.exception('Admin clear database failed')
            return f'✗ Clear failed: {str(e)}'

    @app.callback(
        Output('admin-update-stats-status', 'children'),
        Input('admin-update-features-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def admin_update_meta_features(n_clicks):
        if not n_clicks:
            return ''
        try:
            log.info('Admin requested features & stats update')
            nonlocal color_columns, numeric_columns, feature_groups
            n_features = db.update_meta_features(ch_client)
            meta_features = db.get_meta_features(ch_client, feature_type='meta')
            color_columns = meta_features
            numeric_columns = meta_features
            feature_groups = db.get_feature_groups(ch_client)
            log.info('Admin features & stats update complete: %s features', n_features)
            return f'✓ Updated features & statistics for {n_features} features.'
        except Exception as e:
            log.exception('Admin features & stats update failed')
            return f'✗ Features & stats update failed: {str(e)}'

    @app.callback(
        Output('admin-drop-features-status', 'children'),
        Input('admin-drop-features-btn', 'n_clicks'),
        State('admin-drop-features-dropdown', 'value'),
        prevent_initial_call=True
    )
    def admin_drop_features(n_clicks, selected_features):
        if not n_clicks or not selected_features:
            return ''
        try:
            nonlocal color_columns, numeric_columns, feature_groups
            log.warning('Admin requested drop of %d features', len(selected_features))
            n_dropped = db.drop_features(ch_client, selected_features)
            meta_features = db.get_meta_features(ch_client, feature_type='meta')
            color_columns = meta_features
            numeric_columns = meta_features
            feature_groups = db.get_feature_groups(ch_client)
            return f'✓ Dropped {n_dropped} features. Mutations are running asynchronously.'
        except Exception as e:
            log.exception('Admin drop features failed')
            return f'✗ Drop failed: {str(e)}'

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
        Input('uploaded-point', 'data'),
        Input('hist-quantiles', 'data')
    )

    def update_figure(selected_x, selected_y, selected_color, log_scales, filters, uploaded_point, hist_quantiles):
        t_fig = time.time()
        log.info('Figure update: x=%s y=%s color=%s filters=%s uploaded=%s', selected_x, selected_y, selected_color, len(filters or []), bool(uploaded_point))
        
        if selected_x is None and selected_y is None:
            fig = go.Figure()
            fig.update_layout(template='plotly_white', title='Select X and/or Y to plot')
            log.info('Figure update finished in %.2fs (no axes)', time.time() - t_fig)
            return fig

        features_needed = []
        if selected_x:
            features_needed.append(selected_x)
        if selected_y:
            features_needed.append(selected_y)
        if selected_color:
            features_needed.append(selected_color)
        features_needed.extend(_filter_feature_names(filters))
        features_needed.append('_sample_id')

        df_wide = _get_meta_wide(features_needed)
        if df_wide.empty:
            fig = go.Figure()
            fig.update_layout(template='plotly_white', title='No data available for selected features')
            log.info('Figure update finished in %.2fs (empty data)', time.time() - t_fig)
            return fig

        mask = db.apply_filters_pandas(df_wide, filters)
        df_filtered = df_wide.loc[mask]
        if df_filtered.empty:
            fig = go.Figure()
            fig.update_layout(template='plotly_white', title='No data matches current filters')
            log.info('Figure update finished in %.2fs (no data)', time.time() - t_fig)
            return fig
        
        if df_filtered.empty:
            fig = go.Figure()
            fig.update_layout(template='plotly_white', title='No data available for selected features')
            log.info('Figure update finished in %.2fs (empty data)', time.time() - t_fig)
            return fig

        hist_df = df_filtered
        hist_color_col = None
        hist_color_label = None
        color_info = {'mode': 'none', 'unique_count': 0, 'is_numeric': False, 'is_integer_like': False}
        if selected_color is not None and selected_color in df_filtered.columns:
            hist_df, hist_color_col, hist_color_label, color_info = _prepare_histogram_color_dataframe(
                df_filtered,
                selected_color,
                hist_quantiles,
            )

        if selected_x is not None and selected_y is None:
            if hist_color_col:
                fig = px.histogram(
                    hist_df,
                    x=selected_x,
                    color=hist_color_col,
                    histnorm='probability',
                    template='plotly_white',
                    title=f"Histogram of {selected_x} grouped by {hist_color_label} ({len(hist_df)} points)",
                    opacity=0.7
                )
                fig.update_layout(barmode='overlay')
            else:
                fig = px.histogram(
                    hist_df,
                    x=selected_x,
                    template='plotly_white',
                    title=f"Histogram of {selected_x} ({len(hist_df)} points)"
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
            if hist_color_col:
                fig = px.histogram(
                    hist_df,
                    y=selected_y,
                    color=hist_color_col,
                    histnorm='probability',
                    template='plotly_white',
                    title=f"Histogram of {selected_y} grouped by {hist_color_label} ({len(hist_df)} points)",
                    orientation='h',
                    opacity=0.7
                )
                fig.update_layout(barmode='overlay')
            else:
                fig = px.histogram(
                    hist_df,
                    y=selected_y,
                    template='plotly_white',
                    title=f"Histogram of {selected_y} ({len(hist_df)} points)",
                    orientation='h'
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
            is_categorical = color_info['mode'] == 'categorical'

            if is_categorical:
                freq = df_filtered[selected_color].value_counts()
                df_sorted = df_filtered.copy()
                df_sorted[selected_color] = pd.Categorical(
                    df_sorted[selected_color],
                    categories=freq.index.tolist(),
                    ordered=True
                )
                df_sorted = df_sorted.sort_values(selected_color)

            if color_info['is_numeric'] and not is_categorical:
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
        fig.update_layout(clickmode='event+select', uirevision=uirev, height=800)

        if log_scales and 'x' in log_scales:
            fig.update_xaxes(type='log')
        if log_scales and 'y' in log_scales:
            fig.update_yaxes(type='log')

        if selected_x == 'umap1' and selected_y == 'umap2' and uploaded_point:
            points_list = uploaded_point if isinstance(uploaded_point, list) else [uploaded_point]
            for pt in points_list:
                if not isinstance(pt, dict) or 'umap1' not in pt or 'umap2' not in pt:
                    continue
                pt_label = pt.get('label', 'Uploaded sample')
                fig.add_trace(
                    go.Scatter(
                        x=[pt['umap1']],
                        y=[pt['umap2']],
                        mode='markers',
                        marker=dict(size=18, symbol='star', color='red', line=dict(color='darkred', width=2)),
                        name=pt_label,
                        hovertemplate=f"{pt_label}<br>umap1=%{{x:.3f}}<br>umap2=%{{y:.3f}}<extra></extra>",
                    )
                )
                fig.data[-1].update(showlegend=True)


        log.info('Figure update finished in %.2fs (scatter)', time.time() - t_fig)
        return fig

    @app.callback(
        Output('active-selection-table', 'children'),
        Input('scatter-plot', 'selectedData'),
        Input('x-dropdown', 'value'),
        Input('y-dropdown', 'value'),
        Input('filters-store', 'data'),
        State('detail-columns-dropdown', 'value'),
        State('color-dropdown', 'value')
    )
    def update_active_selection_table(selected_data, selected_x, selected_y, filters, selected_columns, selected_color):
        t_sel = time.time()
        log.info('Active-selection table update: has_selection=%s', bool(selected_data and isinstance(selected_data, dict) and selected_data.get('points')))
        
        # No selection => do not query ClickHouse
        if not (selected_data and isinstance(selected_data, dict) and selected_data.get('points')):
            return ''
        
        if selected_x is None and selected_y is None:
            return ''

        selected_sample_ids = []
        if selected_data is not None and isinstance(selected_data, dict) and 'points' in selected_data and selected_data['points']:
            for p in selected_data['points']:
                cd = p.get('customdata')
                if cd:
                    selected_sample_ids.append(str(cd[0]))

        if not selected_sample_ids:
            return dash_table.DataTable(data=[], columns=[], page_size=50)
        
        features_needed = _effective_detail_features(selected_columns, selected_x, selected_y, selected_color)
        features_needed.extend(_filter_feature_names(filters))
        features_needed.append('_sample_id')

        df_wide = _get_meta_wide(features_needed)
        if df_wide.empty:
            return dash_table.DataTable(data=[], columns=[], page_size=50)

        mask = db.apply_filters_pandas(df_wide, filters)
        df_filtered = df_wide.loc[mask]
        if df_filtered.empty:
            return dash_table.DataTable(data=[], columns=[], page_size=50)

        df_filtered = df_filtered.loc[df_filtered.index.intersection(selected_sample_ids)]
        if df_filtered.empty:
            return dash_table.DataTable(data=[], columns=[], page_size=50)

        df_out = df_filtered.reset_index()
        cols = [{'name': c, 'id': c} for c in df_out.columns]

        log.info('Active-selection table update finished in %.2fs (rows=%s)', time.time() - t_sel, len(df_out))

        return dash_table.DataTable(
            data=df_out.to_dict('records'),
            columns=cols,
            page_size=20,
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
        State('detail-columns-dropdown', 'value'),
        State('color-dropdown', 'value'),
        prevent_initial_call=True
    )
    def export_active_selection_table(n_clicks, selected_data, selected_x, selected_y, filters, selected_columns, selected_color):
        if n_clicks:
            log.info('User requested TSV export (n_clicks=%s)', n_clicks)
        
        if selected_x is None and selected_y is None:
            return None

        selected_sample_ids = []
        if selected_data is not None and isinstance(selected_data, dict) and 'points' in selected_data and selected_data['points']:
            for p in selected_data['points']:
                cd = p.get('customdata')
                if cd:
                    selected_sample_ids.append(str(cd[0]))

        if not selected_sample_ids:
            return None
        
        features_needed = _effective_detail_features(selected_columns, selected_x, selected_y, selected_color)
        features_needed.extend(_filter_feature_names(filters))
        features_needed.append('_sample_id')

        df_wide = _get_meta_wide(features_needed)
        if df_wide.empty:
            return None

        mask = db.apply_filters_pandas(df_wide, filters)
        df_filtered = df_wide.loc[mask]
        if df_filtered.empty:
            return None

        df_filtered = df_filtered.loc[df_filtered.index.intersection(selected_sample_ids)]
        if df_filtered.empty:
            return None

        df_out = df_filtered.reset_index()
        buf = io.StringIO()
        df_out.to_csv(buf, sep='\t', index=False)
        return dict(content=buf.getvalue(), filename='active_selection.tsv', type='text/tab-separated-values')

    @app.callback(
        Output('point-details', 'children'),
        Input('scatter-plot', 'clickData'),
        Input('selected-sample', 'data'),
        Input('detail-columns-dropdown', 'value'),
        State('x-dropdown', 'value'),
        State('y-dropdown', 'value'),
        State('color-dropdown', 'value')
    )
    
    def display_point_details(clickData, selected_sample, selected_columns, selected_x, selected_y, selected_color):
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

        features_needed = _effective_detail_features(selected_columns, selected_x, selected_y, selected_color)
        df_wide = _get_meta_wide(features_needed + ['_sample_id'])
        if df_wide.empty or sample_id not in df_wide.index:
            return "Could not determine selected point (sample id not found in metadata table)."

        df_sample = df_wide.loc[[sample_id], [c for c in features_needed if c in df_wide.columns]]
        
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
        Input('scatter-plot', 'selectedData'),
        Input('selected-sample', 'data'),
        Input('sort-motifs-by-deviation', 'value')
    )
    def display_point_details_core(clickData, selectedData, selected_sample, sort_motifs_value):
        t_core = time.time()
        log.debug('Feature details update (clickData=%s, selectedData=%s)', bool(clickData), bool(selectedData))

        # Collect sample IDs: lasso/box selection takes priority, then click, then stored
        sample_ids = []
        if selectedData and isinstance(selectedData, dict) and selectedData.get('points'):
            for p in selectedData['points']:
                cd = p.get('customdata')
                if cd:
                    sample_ids.append(str(cd[0]))
        if not sample_ids:
            if clickData is not None and 'points' in clickData and clickData['points']:
                point = clickData['points'][0]
                if 'customdata' in point and point['customdata']:
                    sample_ids.append(str(point['customdata'][0]))
            if not sample_ids and selected_sample is not None:
                sample_ids.append(str(selected_sample))

        # Deduplicate while preserving order
        seen = set()
        unique_ids = []
        for sid in sample_ids:
            if sid not in seen:
                seen.add(sid)
                unique_ids.append(sid)
        sample_ids = unique_ids

        if not sample_ids:
            return ""

        n_samples = len(sample_ids)
        log.debug('Feature details for %d sample(s)', n_samples)

        fresh_client = db.get_fresh_client(
            host=getattr(args, 'ch_host', 'localhost'),
            port=getattr(args, 'ch_port', 8123)
        )

        # Get current feature groups and stats from meta_features
        fg = feature_groups
        fszd_col_names = fg['fszd']['col_names']
        fszd_col_names.sort(key=int) #sort numerically otherwise plot makes no sense
        fszd_x = fg['fszd']['x_vals']
        fszd_x.sort(key=int) #sort numerically otherwise plot makes no sense

        csm_col_names = fg['csm']['col_names']
        csm_x = fg['csm']['x_vals']
        p5_col_names = fg['5p']['col_names']
        p5_x = fg['5p']['x_vals']

        all_feature_cols = fszd_col_names + csm_col_names + p5_col_names
        if not all_feature_cols:
            return "No feature groups available to plot for this dataset."

        # Load sample data from meta table
        df_sample_wide = db.lazy_load_meta_samples(fresh_client, sample_ids=sample_ids, features=all_feature_cols)
        if df_sample_wide.empty:
            return f"Selected sample(s) have no feature data."

        # Load feature statistics (mean/std) from meta_features
        df_stats = db.get_feature_stats(fresh_client)
        if not df_stats.empty and 'feature_name' in df_stats.columns:
            df_stats = df_stats.set_index('feature_name')
        else:
            df_stats = pd.DataFrame()

        log.info('Feature details loaded stats (n=%s) samples=%d (prep_total=%.2fs)',
                 len(df_stats) if hasattr(df_stats, '__len__') else 0, n_samples, time.time() - t_core)

        sort_motifs = bool(sort_motifs_value) and ('sort' in sort_motifs_value)

        # Qualitative palette for ≤5 individual sample lines
        _MULTI_COLORS = [
            'rgba(239, 85, 59, 1.0)',    # red
            'rgba(0, 150, 136, 1.0)',     # teal
            'rgba(171, 71, 188, 1.0)',    # purple
            'rgba(255, 160, 0, 1.0)',     # amber
            'rgba(30, 136, 229, 1.0)',    # blue
        ]

        def make_feature_comparison_figure(title, x_vals, col_names, sort_x_by_deviation=False, x_is_categorical=False):
            log.debug('make_feature_comparison_figure: %s (n_samples=%d)', title, n_samples)
            fig_local = go.Figure()

            available = [c for c in col_names if c in df_sample_wide.columns]
            if not available:
                return None

            x_vals_out = list(x_vals[:len(available)]) if len(x_vals) >= len(available) else list(range(len(available)))

            # Population mean/std (blue band) — always shown
            mean_vals = None
            std_vals = None
            if not df_stats.empty:
                mean_vals = df_stats.reindex(available).get('mean')
                std_vals = df_stats.reindex(available).get('std')
                if std_vals is not None:
                    std_vals = std_vals.fillna(0.0)

            # Sort by deviation from mean (use first sample for sort order)
            if sort_x_by_deviation and mean_vals is not None:
                first_sid = df_sample_wide.index[0]
                y_first = pd.to_numeric(df_sample_wide.loc[first_sid, available], errors='coerce')
                dev = (y_first - mean_vals.reindex(available)).abs()
                order = dev.sort_values(ascending=False).index.tolist()
                x_by_col = dict(zip(available, x_vals_out))
                x_vals_out = [x_by_col.get(col, str(col)) for col in order]
                available = order
                mean_vals = mean_vals.reindex(available)
                std_vals = std_vals.reindex(available)

            # Draw population mean ± SD band (blue)
            if mean_vals is not None and std_vals is not None:
                m = mean_vals.values
                s = std_vals.values
                if len(m) == len(x_vals_out) and len(s) == len(x_vals_out):
                    upper = m + s
                    lower = m - s
                    fig_local.add_trace(
                        go.Scatter(
                            x=list(x_vals_out) + list(x_vals_out)[::-1],
                            y=list(upper) + list(lower[::-1]),
                            fill='toself',
                            fillcolor='rgba(99, 110, 250, 0.15)',
                            line={'color': 'rgba(255,255,255,0)'},
                            hoverinfo='skip',
                            name='Population Mean ± 1 SD'
                        )
                    )
                    fig_local.add_trace(
                        go.Scatter(
                            x=x_vals_out,
                            y=m,
                            mode='lines',
                            line={'color': 'rgba(99, 110, 250, 0.9)', 'width': 2},
                            name='Population Mean'
                        )
                    )

            # ── Plot samples ──
            present_ids = [sid for sid in sample_ids if sid in df_sample_wide.index]

            if n_samples <= 5:
                # Individual lines for each sample
                for idx, sid in enumerate(present_ids):
                    y_vals = pd.to_numeric(df_sample_wide.loc[sid, available], errors='coerce')
                    color = _MULTI_COLORS[idx % len(_MULTI_COLORS)]
                    fig_local.add_trace(
                        go.Scatter(
                            x=x_vals_out,
                            y=y_vals.values,
                            mode='lines',
                            line={'color': color, 'width': 2},
                            name=sid
                        )
                    )
            else:
                # >5 samples: compute selection mean ± SD and plot in red
                all_vals = []
                for sid in present_ids:
                    row = pd.to_numeric(df_sample_wide.loc[sid, available], errors='coerce')
                    all_vals.append(row.values)
                if all_vals:
                    arr = np.array(all_vals, dtype=float)
                    sel_mean = np.nanmean(arr, axis=0)
                    sel_std = np.nanstd(arr, axis=0)
                    sel_upper = sel_mean + sel_std
                    sel_lower = sel_mean - sel_std
                    fig_local.add_trace(
                        go.Scatter(
                            x=list(x_vals_out) + list(x_vals_out)[::-1],
                            y=list(sel_upper) + list(sel_lower[::-1]),
                            fill='toself',
                            fillcolor='rgba(239, 85, 59, 0.15)',
                            line={'color': 'rgba(255,255,255,0)'},
                            hoverinfo='skip',
                            name=f'Selection Mean ± 1 SD (n={len(present_ids)})'
                        )
                    )
                    fig_local.add_trace(
                        go.Scatter(
                            x=x_vals_out,
                            y=sel_mean,
                            mode='lines',
                            line={'color': 'rgba(239, 85, 59, 1.0)', 'width': 2},
                            name=f'Selection Mean (n={len(present_ids)})'
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

        if fszd_col_names:
            fig_fszd = make_feature_comparison_figure('Fragment size distribution', fszd_x, fszd_col_names)
            if fig_fszd:
                fig_fszd.update_xaxes(title_text='Fragment size')
                fig_fszd.update_yaxes(title_text='Value')
                children.append(dcc.Graph(figure=fig_fszd, config={'responsive': True}))

        if csm_col_names:
            fig_csm = make_feature_comparison_figure('CSM features', csm_x, csm_col_names, sort_x_by_deviation=sort_motifs, x_is_categorical=True)
            if fig_csm:
                fig_csm.update_xaxes(title_text='Motif')
                fig_csm.update_yaxes(title_text='Value')
                children.append(dcc.Graph(figure=fig_csm, config={'responsive': True}))

        if p5_col_names:
            fig_p5 = make_feature_comparison_figure("5' features", p5_x, p5_col_names, sort_x_by_deviation=sort_motifs, x_is_categorical=True)
            if fig_p5:
                fig_p5.update_xaxes(title_text='Motif')
                fig_p5.update_yaxes(title_text='Value')
                children.append(dcc.Graph(figure=fig_p5, config={'responsive': True}))

        if not children:
            return "No feature data available to plot for selected sample(s)."

        return html.Div(children)

    # hg38 chromosome lengths for proportional sizing
    HG38_CHROM_SIZES = {
        'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559,
        'chr4': 190214555, 'chr5': 181538259, 'chr6': 170805979,
        'chr7': 159345973, 'chr8': 145138636, 'chr9': 138394717,
        'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309,
        'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189,
        'chr16': 90338345, 'chr17': 83257441, 'chr18': 80373285,
        'chr19': 58617616, 'chr20': 64444167, 'chr21': 46709983,
        'chr22': 50818468, 'chrX': 156040895, 'chrY': 57227415,
    }
    CHROM_ORDER = [
        'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
        'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
        'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22',
        'chrX', 'chrY',
    ]

    @app.callback(
        Output('point-details-bincount', 'children'),
        Input('scatter-plot', 'clickData'),
        Input('selected-sample', 'data'),
        Input('zscore-threshold', 'value')
    )
    def display_bincount_zscore(clickData, selected_sample, zscore_threshold):
        sample_id = None
        if clickData is not None and 'points' in clickData and clickData['points']:
            point = clickData['points'][0]
            if 'customdata' in point and point['customdata']:
                sample_id = str(point['customdata'][0])
        if sample_id is None and selected_sample is not None:
            sample_id = str(selected_sample)
        if sample_id is None:
            return ""

        fg = feature_groups
        bin_info = fg.get('bincount', {})
        bin_cols = bin_info.get('col_names', [])
        if not bin_cols:
            return ""

        zscore_threshold = float(zscore_threshold) if zscore_threshold else 3.0

        fresh_client = db.get_fresh_client(
            host=getattr(args, 'ch_host', 'localhost'),
            port=getattr(args, 'ch_port', 8123)
        )

        # Load sample values for bincount features
        df_sample = db.lazy_load_meta_samples(fresh_client, sample_ids=[sample_id], features=bin_cols)
        if df_sample.empty:
            return f"No bincount data for sample {sample_id}."

        # Load stats for bincount features
        df_stats = db.get_feature_stats(fresh_client)
        if df_stats.empty:
            return "No feature statistics available for bincount Z-score plot."
        df_stats = df_stats.set_index('feature_name')

        # Build a DataFrame with chrom, start, end, value, mean, std, zscore
        bin_chroms = bin_info.get('chromosomes', [])
        bin_starts = bin_info.get('starts', [])
        bin_ends = bin_info.get('ends', [])

        records = []
        row = df_sample.iloc[0]
        for i, col in enumerate(bin_cols):
            if col not in df_sample.columns:
                continue
            val = pd.to_numeric(row.get(col), errors='coerce')
            if pd.isna(val):
                continue
            mean = df_stats.at[col, 'mean'] if col in df_stats.index else None
            std = df_stats.at[col, 'std'] if col in df_stats.index else None
            if mean is None or std is None or pd.isna(mean) or pd.isna(std) or std == 0:
                continue
            z = (val - mean) / std
            records.append({
                'chrom': bin_chroms[i],
                'start': bin_starts[i],
                'end': bin_ends[i],
                'mid': (bin_starts[i] + bin_ends[i]) / 2,
                'value': val,
                'mean': mean,
                'std': std,
                'zscore': z,
                'col_name': col,
            })

        if not records:
            return "Could not compute Z-scores (no valid bincount stats)."

        df_bin = pd.DataFrame(records)

        # Determine which chromosomes are present and in canonical order
        present_chroms = [c for c in CHROM_ORDER if c in df_bin['chrom'].unique()]
        if not present_chroms:
            # Try without 'chr' prefix
            present_chroms = sorted(df_bin['chrom'].unique(), key=lambda c: CHROM_ORDER.index(c) if c in CHROM_ORDER else 999)
        if not present_chroms:
            return "No recognized chromosomes in bincount data."

        # Compute cumulative offsets for ideogram layout
        chrom_sizes = {}
        for c in present_chroms:
            chrom_sizes[c] = HG38_CHROM_SIZES.get(c, df_bin.loc[df_bin['chrom'] == c, 'end'].max())

        total_genome = sum(chrom_sizes[c] for c in present_chroms)
        cum_offset = {}
        running = 0
        for c in present_chroms:
            cum_offset[c] = running
            running += chrom_sizes[c]

        df_bin['genome_pos'] = df_bin.apply(lambda r: cum_offset.get(r['chrom'], 0) + r['mid'], axis=1)

        # Build the figure
        fig = go.Figure()

        # Alternating background rectangles per chromosome
        for idx_c, c in enumerate(present_chroms):
            x0 = cum_offset[c]
            x1 = x0 + chrom_sizes[c]
            if idx_c % 2 == 0:
                fig.add_vrect(x0=x0, x1=x1, fillcolor='rgba(200,200,200,0.15)',
                              line_width=0, layer='below')

        # Threshold bands
        fig.add_hline(y=zscore_threshold, line_dash='dash', line_color='rgba(220,53,69,0.5)', line_width=1)
        fig.add_hline(y=-zscore_threshold, line_dash='dash', line_color='rgba(220,53,69,0.5)', line_width=1)
        fig.add_hline(y=0, line_color='rgba(0,0,0,0.3)', line_width=0.5)

        # Split into normal and outlier points
        normal = df_bin[df_bin['zscore'].abs() <= zscore_threshold]
        outlier = df_bin[df_bin['zscore'].abs() > zscore_threshold]

        if not normal.empty:
            fig.add_trace(go.Scattergl(
                x=normal['genome_pos'],
                y=normal['zscore'],
                mode='markers',
                marker=dict(size=3, color='rgba(99,110,250,0.5)'),
                name='Z-score',
                hovertemplate='%{customdata[0]}<br>Z=%{y:.2f}<extra></extra>',
                customdata=normal[['col_name']].values,
            ))

        if not outlier.empty:
            fig.add_trace(go.Scattergl(
                x=outlier['genome_pos'],
                y=outlier['zscore'],
                mode='markers',
                marker=dict(size=5, color='rgba(220,53,69,0.9)'),
                name=f'|Z| > {zscore_threshold}',
                hovertemplate='%{customdata[0]}<br>Z=%{y:.2f}<extra></extra>',
                customdata=outlier[['col_name']].values,
            ))

        # Chromosome-wide Stouffer Z-score lines
        stouffer_shown_legend = {'blue': False, 'red': False}
        for c in present_chroms:
            chrom_zscores = df_bin.loc[df_bin['chrom'] == c, 'zscore'].values
            n = len(chrom_zscores)
            if n == 0:
                continue
            stouffer_z = chrom_zscores.sum() / np.sqrt(n)
            x0 = cum_offset[c]
            x1 = x0 + chrom_sizes[c]
            is_outlier = abs(stouffer_z) >= zscore_threshold
            color = 'rgba(220,53,69,0.8)' if is_outlier else 'rgba(31,119,180,0.8)'
            legend_key = 'red' if is_outlier else 'blue'
            show_legend = not stouffer_shown_legend[legend_key]
            stouffer_shown_legend[legend_key] = True
            label = c.replace('chr', '') if c.startswith('chr') else c
            fig.add_trace(go.Scattergl(
                x=[x0, x1],
                y=[stouffer_z, stouffer_z],
                mode='lines',
                line=dict(color=color, width=2),
                name=f'Stouffer |Z| {"≥" if is_outlier else "<"} {zscore_threshold}',
                showlegend=show_legend,
                hovertemplate=f'chr{label} Stouffer Z={stouffer_z:.2f} (n={n})<extra></extra>',
            ))

        # Chromosome labels on x-axis
        tick_vals = []
        tick_text = []
        for c in present_chroms:
            mid = cum_offset[c] + chrom_sizes[c] / 2
            tick_vals.append(mid)
            label = c.replace('chr', '') if c.startswith('chr') else c
            tick_text.append(label)

        fig.update_layout(
            template='plotly_white',
            height=350,
            margin=dict(l=50, r=20, t=40, b=50),
            title=f'Genome-wide bincount Z-scores (threshold ±{zscore_threshold})',
            xaxis=dict(
                title='Chromosome',
                tickvals=tick_vals,
                ticktext=tick_text,
                range=[0, total_genome],
            ),
            yaxis=dict(title='Z-score'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            showlegend=True,
        )

        return dcc.Graph(figure=fig, config={'responsive': True})

    app.run(debug=False, host="0.0.0.0", port=8050)