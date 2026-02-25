import pandas as pd 
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dash.exceptions import PreventUpdate

# Example dataframe
# df = pd.read_csv(...)
# Must contain columns: 'x', 'y'

def explore(args):

    dfcore = pd.read_csv(args.core, sep='\t', index_col=0)
    dfmeta = pd.read_csv(args.meta, sep='\t')
    dfmeta = dfmeta.drop_duplicates(subset='filename').set_index('filename')
    dfmeta['_sample_id'] = dfmeta.index.astype(str)

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
    
    app = dash.Dash("Fragmentome explorer")

    app.layout = html.Div([
        html.H3("Fragmentome explorer"),

        dcc.Store(id='selected-sample', data=None),
        
        dcc.Loading(
            id="loading-icon",
            children=[html.Div(id='loading-status', children="Ready", style={'padding': '10px', 'color': 'green'})],
            type="default"
        ),

        html.Div([
            html.Div([
                html.Label("Color by:"),
                dcc.Dropdown(
                    id='color-dropdown',
                    options=[{'label': col, 'value': col} for col in color_columns],
                    value=color_columns[0],
                    clearable=False
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
        )
    ])

    @app.callback(
        Output('selected-sample', 'data'),
        Input('scatter-plot', 'clickData')
    )
    def update_selected_sample(clickData):
        if clickData is None or 'points' not in clickData or not clickData['points']:
            return None

        point = clickData['points'][0]
        if 'customdata' not in point or not point['customdata']:
            return None

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
        Input('color-dropdown', 'value'),
        Input('filter-dropdown', 'value'),
        Input('filter-range', 'value'),
        Input('selected-sample', 'data')
    )

    def update_figure(selected_color, filter_column, filter_range, selected_sample):
        # Start with full dataframe
        df_filtered = dfmeta.copy()
        
        # Apply filters only for numeric columns and only if range is set to non-default values
        if filter_column is not None and pd.api.types.is_numeric_dtype(dfmeta[filter_column]):
            if filter_range is not None and filter_range != [0, 1]:
                # Apply numeric range filter (only if not default values)
                min_val, max_val = filter_range
                df_filtered = df_filtered[
                    (df_filtered[filter_column] >= min_val) & 
                    (df_filtered[filter_column] <= max_val)
                ]
        
        # Sort by frequency for categorical variables
        # Treat columns as categorical if they're non-numeric OR have ≤25 distinct integer-like values
        is_categorical = not pd.api.types.is_numeric_dtype(df_filtered[selected_color]) or \
                         (pd.api.types.is_numeric_dtype(df_filtered[selected_color]) and 
                          len(df_filtered[selected_color].dropna().unique()) <= 25 and
                          all(val.is_integer() if isinstance(val, float) else True 
                              for val in df_filtered[selected_color].dropna().unique()))
        
        if is_categorical:
            # For categorical variables, sort by frequency (most frequent first)
            freq = df_filtered[selected_color].value_counts()
            df_sorted = df_filtered.copy()
            df_sorted[selected_color] = pd.Categorical(
                df_sorted[selected_color],
                categories=freq.index.tolist(),
                ordered=True
            )
            df_sorted = df_sorted.sort_values(selected_color)
        else:
            df_sorted = df_filtered

        # Automatic handling of categorical vs numeric
        if pd.api.types.is_numeric_dtype(df_filtered[selected_color]) and not is_categorical:
            fig = px.scatter(
                df_sorted,
                x='x',
                y='y',
                color=selected_color,
                color_continuous_scale='Viridis',
                template="plotly_white",
                hover_data=['x', 'y'],
                custom_data=['_sample_id'],
                labels={selected_color: selected_color},
                title=f"Scatter plot colored by {selected_color} ({len(df_filtered)} points)",
                # width=1000,
                # height=800
            )
        else:
            fig = px.scatter(
                df_sorted,
                x='x',
                y='y',
                color=selected_color,
                template="plotly_white",
                hover_data=['x', 'y'],
                custom_data=['_sample_id'],
                labels={selected_color: selected_color},
                title=f"Scatter plot colored by {selected_color} ({len(df_filtered)} points)",
                # width=1000,
                # height=800
            )

        uirev = f"{selected_color}|{filter_column}|{filter_range}"
        fig.update_layout(clickmode='event+select', uirevision=uirev)

        if selected_sample is not None and selected_sample in df_sorted.index.astype(str).tolist():
            sel_row = df_sorted.loc[selected_sample]
            fig.add_trace(
                go.Scatter(
                    x=[sel_row['x']],
                    y=[sel_row['y']],
                    mode='markers',
                    marker=dict(
                        size=14,
                        color='rgba(0,0,0,0)',
                        line=dict(color='rgba(239, 85, 59, 1.0)', width=3)
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )

        return fig

    @app.callback(
        Output('point-details', 'children'),
        Input('scatter-plot', 'clickData'),
        Input('detail-columns-dropdown', 'value'),
        Input('filter-dropdown', 'value'),
        Input('filter-range', 'value')
    )
    
    def display_point_details(clickData, selected_columns, filter_column, filter_range):
        if clickData is None or 'points' not in clickData or not clickData['points']:
            return "Click a point in the scatter plot to see its values here."

        point = clickData['points'][0]
        sample_id = None
        if 'customdata' in point and point['customdata']:
            sample_id = str(point['customdata'][0])
        if sample_id is None or sample_id not in dfmeta.index.astype(str).tolist():
            return "Could not determine selected point."

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
        Input('scatter-plot', 'clickData')
    )
    def display_point_details_core(clickData):
        if clickData is None or 'points' not in clickData or not clickData['points']:
            return ""

        point = clickData['points'][0]
        if 'customdata' not in point or not point['customdata']:
            return ""

        sample_id = str(point['customdata'][0])
        if sample_id not in dfcore.index.astype(str).tolist():
            return ""

        if not core_col_names:
            return ""

        y_sample = pd.to_numeric(dfcore.loc[sample_id, core_col_names], errors='coerce')

        fig = go.Figure()

        mean_vals = core_mean.values
        std_vals = core_std.values
        x_vals = core_x

        if len(mean_vals) == len(x_vals) and len(std_vals) == len(x_vals):
            upper = mean_vals + std_vals
            lower = mean_vals - std_vals
            fig.add_trace(
                go.Scatter(
                    x=x_vals + x_vals[::-1],
                    y=list(upper) + list(lower[::-1]),
                    fill='toself',
                    fillcolor='rgba(99, 110, 250, 0.15)',
                    line={'color': 'rgba(255,255,255,0)'},
                    hoverinfo='skip',
                    name='Mean ± 1 SD'
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=mean_vals,
                    mode='lines',
                    line={'color': 'rgba(99, 110, 250, 0.9)', 'width': 2},
                    name='Mean'
                )
            )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_sample.values,
                mode='lines',
                line={'color': 'rgba(239, 85, 59, 1.0)', 'width': 2},
                name=sample_id
            )
        )

        fig.update_layout(
            template='plotly_white',
            margin=dict(l=40, r=20, t=40, b=40),
            height=360,
            legend=dict(orientation='h')
        )
        fig.update_xaxes(title_text='Core position')
        fig.update_yaxes(title_text='Value')

        return dcc.Graph(figure=fig, config={'responsive': True})

    app.run(debug=True, host="0.0.0.0", port=8050)