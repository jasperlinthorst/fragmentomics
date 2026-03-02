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

from cfstats import fszd, csm, fpends

# Example dataframe
# df = pd.read_csv(...)
# Must contain columns: 'x', 'y'

def explore(args):

    dfcore = pd.read_csv(args.core, sep='\t', index_col=0)

    dfmeta = pd.read_csv(args.meta, sep='\t')
    dfmeta = dfmeta.drop_duplicates(subset='filename').set_index('filename')
    dfcore.index = dfcore.index.astype(str)
    dfmeta.index = dfmeta.index.astype(str)
    dfmeta['_sample_id'] = dfmeta.index

    numeric_columns = [c for c in dfmeta.columns if pd.api.types.is_numeric_dtype(dfmeta[c])]
    default_x_col = None
    default_y_col = None

    mapping = None
    if getattr(args, 'mapping', None):
        mapping = pickle.load(open(args.mapping, 'rb'))
        reducer = mapping[0]
        embedding = mapping[1]
        mapping_k = mapping[2]

    color_columns = []
    
    # Convert float64 columns to int when possible and distinct values ≤ 25
    for col in dfmeta.select_dtypes(include=['float64', 'int64', 'object']).columns:
        if col in ['x', 'y']:
            continue
        distinct_vals = dfmeta[col].dropna().unique()
        if len(distinct_vals) <= 25:
            # # Check if all values are integers (no decimal part)
            # if all(val.is_integer() for val in distinct_vals):
            dfmeta[col] = dfmeta[col].astype(str)  # nullable integer type
        color_columns.append(col)

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

    dfcore_values = dfcore[core_col_names].apply(pd.to_numeric, errors='coerce') if core_col_names else pd.DataFrame(index=dfcore.index)
    core_mean = dfcore_values.mean(axis=0) if not dfcore_values.empty else pd.Series(dtype=float)
    core_std = dfcore_values.std(axis=0) if not dfcore_values.empty else pd.Series(dtype=float)

    csm_col_names = [c for c in dfcore.columns if str(c).endswith('-csm')]
    csm_x = [str(c)[:-4] for c in csm_col_names]

    p5_col_names = [c for c in dfcore.columns if str(c).endswith('-5p')]
    p5_x = [str(c)[:-3] for c in p5_col_names]
    
    app = dash.Dash("Fragmentome explorer")

    app.layout = html.Div([
        html.H3("Fragmentome explorer"),

        dcc.Store(id='selected-sample', data=None),
        dcc.Store(id='uploaded-point', data=None),
        
        dcc.Loading(
            id="loading-icon",
            children=[html.Div(id='loading-status', children="Ready", style={'padding': '10px', 'color': 'green', 'fontWeight': 'bold'})],
            type="default"
        ),

        dcc.Upload(
            id='upload-alignment',
            children=html.Div(['Drag and Drop or ', html.A('Select BAM/SAM/CRAM')]),
            max_size=200*1024*1024, # 200MB
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
                dcc.Dropdown(
                    id='filter-dropdown',
                    options=[{'label': col, 'value': col} for col in dfmeta.columns],
                    value=None,
                    clearable=True,
                    placeholder="Select column to filter"
                )
            ], style={'width': '20%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                html.Label("Filter range:"),
                dcc.RangeSlider(
                    id='filter-range',
                    min=0,
                    max=1,
                    step=0.01,
                    value=[0, 1],
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '20%', 'display': 'inline-block', 'padding': '10px'})
        ]),

        dcc.Loading(
            id="loading-scatter",
            children=[dcc.Graph(id='scatter-plot')],
            style={'padding': '10px', 'width': '100%', 'height': '100%'},
            type="graph"
        ),

        dcc.Dropdown(
            id='detail-columns-dropdown',
            options=[{'label': col, 'value': col} for col in dfmeta.columns],
            value=list(dfmeta.columns),
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
                html.Div(
                    id='point-details-core',
                    style={'flex': '2 1 0', 'minWidth': '360px', 'padding': '10px'}
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
            return None, '', current_x, current_y, 'Ready'

        # Show processing status immediately
        if isinstance(contents, (list, tuple)):
            if not contents:
                return None, 'Upload failed: empty contents.', current_x, current_y, 'Error: Empty upload'
            contents = contents[0]

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

        # print("Contents:", contents)
        print("Filename:", filename)

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

            print(f"Uploaded file saved to: {out_path}")
            
            # Update status to show processing
            print(f"Computing features for uploaded file...", mapping_k)
            
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
            print("Xfszd shape:", Xfszd, Xfszd.shape)
            Xcsm = np.array(csm.cleavesitemotifs(upload_args, cmdline=False))
            print("Xcsm shape:", Xcsm, Xcsm.shape)
            Xsem = np.array(fpends._5pends(upload_args, cmdline=False))
            print("Xsem shape:", Xsem, Xsem.shape)
            f = np.concatenate((Xfszd, Xcsm, Xsem), axis=1)
            
            print(f"Features computed, shape: {f.shape}, reducing...")
            
            fp = reducer.transform(f)
            
            print(f"Reduced features, shape: {fp.shape}")

            x_new = float(fp[0, 0])
            y_new = float(fp[0, 1])
            label = filename if filename else os.path.basename(out_path)

            print(f"Processing uploaded file: {filename}...")
            
            return {
                'umap1': x_new,
                'umap2': y_new,
                'label': label,
            }, f'Processed: {label} (umap1={x_new:.3f}, umap2={y_new:.3f})', 'umap1', 'umap2', f'✓ Processed: {label}'
        except Exception as e:
            return None, f'Upload failed: {str(e)}', current_x, current_y, f'Error: {str(e)}'
        finally:
            pass

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

        return str(point['customdata'][0])

    @app.callback(
        Output('filter-range', 'min'),
        Output('filter-range', 'max'),
        Output('filter-range', 'value'),
        Output('filter-range', 'marks'),
        Input('filter-dropdown', 'value')
    )

    def update_filter_controls(filter_column):
        if filter_column is None:
            return 0, 1, [0, 1], {0: '0', 1: '1'}
        
        # Only show filter controls for numeric columns
        is_numeric = pd.api.types.is_numeric_dtype(dfmeta[filter_column])
        
        if is_numeric:
            min_val = dfmeta[filter_column].min()
            max_val = dfmeta[filter_column].max()
            
            return min_val, max_val, [min_val, max_val], {int(min_val): str(min_val), int(max_val): str(max_val)}
        else:
            # For non-numeric columns, hide filter controls
            return 0, 1, [0, 1], {0: '0', 1: '1'}

    @app.callback(
        Output('scatter-plot', 'figure'),
        Input('x-dropdown', 'value'),
        Input('y-dropdown', 'value'),
        Input('color-dropdown', 'value'),
        Input('log-scales', 'value'),
        Input('filter-dropdown', 'value'),
        Input('filter-range', 'value'),
        Input('selected-sample', 'data'),
        Input('uploaded-point', 'data')
    )

    def update_figure(selected_x, selected_y, selected_color, log_scales, filter_column, filter_range, selected_sample, uploaded_point):
        # Start with full dataframe
        df_filtered = dfmeta.copy()

        if selected_x is None and selected_y is None:
            fig = go.Figure()
            fig.update_layout(template='plotly_white', title='Select X and/or Y to plot')
            return fig
        
        # Apply filters only for numeric columns and only if range is set to non-default values
        if filter_column is not None and pd.api.types.is_numeric_dtype(dfmeta[filter_column]):
            if filter_range is not None and filter_range != [0, 1]:
                # Apply numeric range filter (only if not default values)
                min_val, max_val = filter_range
                df_filtered = df_filtered[
                    (df_filtered[filter_column] >= min_val) & 
                    (df_filtered[filter_column] <= max_val)
                ]

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
            fig.update_layout(uirevision=f"hx|{selected_x}|{selected_color}|{filter_column}|{filter_range}")
            fig.update_layout(clickmode='event+select')
            if log_scales and 'x' in log_scales:
                fig.update_xaxes(type='log')
            if log_scales and 'y' in log_scales:
                fig.update_yaxes(type='log')
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
            fig.update_layout(uirevision=f"vy|{selected_y}|{selected_color}|{filter_column}|{filter_range}")
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

        uirev = f"{selected_x}|{selected_y}|{selected_color}|{filter_column}|{filter_range}"
        fig.update_layout(clickmode='event+select', uirevision=uirev)

        if log_scales and 'x' in log_scales:
            fig.update_xaxes(type='log')
        if log_scales and 'y' in log_scales:
            fig.update_yaxes(type='log')

        if selected_sample is not None and selected_sample in df_sorted.index.astype(str).tolist():
            sel_row = df_sorted.loc[selected_sample]
            fig.add_trace(
                go.Scatter(
                    x=[sel_row[selected_x]],
                    y=[sel_row[selected_y]],
                    mode='markers',
                    marker=dict(
                        size=14,
                        color='rgba(0,0,0,0)',
                        line=dict(color='rgba(239, 85, 59, 1.0)', width=3)
                    ),
                    customdata=[selected_sample],
                    showlegend=False,
                    hoverinfo='skip'
                )
            )

        if selected_x == 'umap1' and selected_y == 'umap2' and uploaded_point is not None and 'umap1' in uploaded_point and 'umap2' in uploaded_point:
            fig.add_trace(
                go.Scatter(
                    x=[uploaded_point['umap1']],
                    y=[uploaded_point['umap2']],
                    mode='markers',
                    marker=dict(size=18, symbol='star', color='red', line=dict(color='darkred', width=2)),
                    name='Uploaded sample',
                    hovertemplate=f"{uploaded_point.get('label','Uploaded')}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<extra></extra>",
                )
            )

        return fig

    @app.callback(
        Output('active-selection-table', 'children'),
        Input('scatter-plot', 'selectedData'),
        Input('x-dropdown', 'value'),
        Input('y-dropdown', 'value'),
        Input('filter-dropdown', 'value'),
        Input('filter-range', 'value')
    )
    def update_active_selection_table(selected_data, selected_x, selected_y, filter_column, filter_range):
        df_filtered = dfmeta.copy()

        if selected_x is None and selected_y is None:
            return ''

        if filter_column is not None and pd.api.types.is_numeric_dtype(dfmeta[filter_column]):
            if filter_range is not None and filter_range != [0, 1]:
                min_val, max_val = filter_range
                df_filtered = df_filtered[
                    (df_filtered[filter_column] >= min_val) &
                    (df_filtered[filter_column] <= max_val)
                ]

        if selected_data is not None and isinstance(selected_data, dict) and 'points' in selected_data and selected_data['points']:
            sample_ids = []
            for p in selected_data['points']:
                cd = p.get('customdata')
                if cd:
                    sample_ids.append(str(cd[0]))
            if sample_ids:
                df_filtered = df_filtered.loc[df_filtered.index.astype(str).isin(sample_ids)]

        df_head = df_filtered.head(50).reset_index()
        cols = [{'name': c, 'id': c} for c in df_head.columns]

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
        State('filter-dropdown', 'value'),
        State('filter-range', 'value'),
        prevent_initial_call=True
    )
    def export_active_selection_table(n_clicks, selected_data, selected_x, selected_y, filter_column, filter_range):
        df_filtered = dfmeta.copy()

        if selected_x is None and selected_y is None:
            return None

        if filter_column is not None and pd.api.types.is_numeric_dtype(dfmeta[filter_column]):
            if filter_range is not None and filter_range != [0, 1]:
                min_val, max_val = filter_range
                df_filtered = df_filtered[
                    (df_filtered[filter_column] >= min_val) &
                    (df_filtered[filter_column] <= max_val)
                ]

        if selected_data is not None and isinstance(selected_data, dict) and 'points' in selected_data and selected_data['points']:
            sample_ids = []
            for p in selected_data['points']:
                cd = p.get('customdata')
                if cd:
                    sample_ids.append(str(cd[0]))
            if sample_ids:
                df_filtered = df_filtered.loc[df_filtered.index.astype(str).isin(sample_ids)]

        df_out = df_filtered.head(50).reset_index()
        buf = io.StringIO()
        df_out.to_csv(buf, sep='\t', index=False)
        return dict(content=buf.getvalue(), filename='active_selection.tsv', type='text/tab-separated-values')

    @app.callback(
        Output('point-details', 'children'),
        Input('scatter-plot', 'clickData'),
        Input('selected-sample', 'data'),
        Input('detail-columns-dropdown', 'value'),
        Input('filter-dropdown', 'value'),
        Input('filter-range', 'value')
    )
    
    def display_point_details(clickData, selected_sample, selected_columns, filter_column, filter_range):
        print("display_point_details")
        sample_id = None
        if clickData is not None and 'points' in clickData and clickData['points']:
            point = clickData['points'][0]
            if 'customdata' in point and point['customdata']:
                sample_id = str(point['customdata'][0])

        if sample_id is None and selected_sample is not None:
            sample_id = str(selected_sample)

        if sample_id is None:
            return "Click a point in the scatter plot to see its values here."

        if sample_id not in dfmeta.index:
            return "Could not determine selected point (sample id not found in metadata table)."

        print(sample_id)
        row = dfmeta.loc[sample_id]
        print(row.shape)
        # Use selected columns, fallback to all columns if none selected
        columns_to_show = selected_columns if selected_columns else list(dfmeta.columns)

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
        Input('scatter-plot', 'selectedData'),
        Input('filter-dropdown', 'value'),
        Input('filter-range', 'value'),
        Input('sort-motifs-by-deviation', 'value')
    )
    def display_point_details_core(clickData, selected_sample, selected_data, filter_column, filter_range, sort_motifs_value):
        print("display_point_details_core", clickData)
        print("click",clickData)
        sample_id = None
        if clickData is not None and 'points' in clickData and clickData['points']:
            point = clickData['points'][0]
            if 'customdata' in point and point['customdata']:
                sample_id = str(point['customdata'][0])

        if sample_id is None and selected_sample is not None:
            sample_id = str(selected_sample)

        if sample_id is None:
            return ""

        print("sample_id", sample_id)

        if sample_id not in dfcore.index:
            print(f"Selected sample {sample_id} is not present in the core feature table.")
            return f"Selected sample {sample_id} is not present in the core feature table."

        print("filtereing")

        df_filtered = dfmeta.copy()
        if filter_column is not None and pd.api.types.is_numeric_dtype(dfmeta[filter_column]):
            if filter_range is not None and filter_range != [0, 1]:
                min_val, max_val = filter_range
                df_filtered = df_filtered[
                    (df_filtered[filter_column] >= min_val) &
                    (df_filtered[filter_column] <= max_val)
                ]

        print("cloudids")

        cloud_ids = df_filtered.index.astype(str).tolist()
        dfcoreids = set(dfcore.index.astype(str).tolist())
        cloud_ids_core = [sid for sid in cloud_ids if sid in dfcoreids]

        print("cloudids done")

        sort_motifs = bool(sort_motifs_value) and ('sort' in sort_motifs_value)

        def make_feature_comparison_figure(title, x_vals, y_sample, df_cloud_values, sort_x_by_deviation=False, x_is_categorical=False):
            print("make_feature_comparison_figure", title)
            fig_local = go.Figure()

            x_vals_out = list(x_vals)
            y_sample_out = y_sample.copy()

            if df_cloud_values is not None and not df_cloud_values.empty:
                mean_vals_local = df_cloud_values.mean(axis=0)
                std_vals_local = df_cloud_values.std(axis=0)

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
            y_sample_core = pd.to_numeric(dfcore.loc[sample_id, core_col_names], errors='coerce')
            cloud_core_values = None
            if cloud_ids_core and core_col_names:
                cloud_core_values = dfcore.loc[cloud_ids_core, core_col_names].apply(pd.to_numeric, errors='coerce')
            fig_core = make_feature_comparison_figure('Fragment size distribution (core)', core_x, y_sample_core, cloud_core_values)
            fig_core.update_xaxes(title_text='Core position')
            fig_core.update_yaxes(title_text='Value')
            children.append(dcc.Graph(figure=fig_core, config={'responsive': True}))

        if csm_col_names:
            y_sample_csm = pd.to_numeric(dfcore.loc[sample_id, csm_col_names], errors='coerce')
            cloud_csm_values = None
            if cloud_ids_core:
                cloud_csm_values = dfcore.loc[cloud_ids_core, csm_col_names].apply(pd.to_numeric, errors='coerce')
            fig_csm = make_feature_comparison_figure('CSM features', csm_x, y_sample_csm, cloud_csm_values, sort_x_by_deviation=sort_motifs, x_is_categorical=True)
            fig_csm.update_xaxes(title_text='Motif')
            fig_csm.update_yaxes(title_text='Value')
            children.append(dcc.Graph(figure=fig_csm, config={'responsive': True}))

        if p5_col_names:
            y_sample_p5 = pd.to_numeric(dfcore.loc[sample_id, p5_col_names], errors='coerce')
            cloud_p5_values = None
            if cloud_ids_core:
                cloud_p5_values = dfcore.loc[cloud_ids_core, p5_col_names].apply(pd.to_numeric, errors='coerce')
            fig_p5 = make_feature_comparison_figure("5' features", p5_x, y_sample_p5, cloud_p5_values, sort_x_by_deviation=sort_motifs, x_is_categorical=True)
            fig_p5.update_xaxes(title_text='Motif')
            fig_p5.update_yaxes(title_text='Value')
            children.append(dcc.Graph(figure=fig_p5, config={'responsive': True}))

        if not children:
            return "No core/CSM/5p features available to plot for this dataset."

        return html.Div(children)

    app.run(debug=False, host="0.0.0.0", port=8050)