"""
Use dash to produce a visualization for an asset-level drill-down
Dash app then migrated to React / Flask
"""



from datetime import datetime
import dash, dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash_table.Format import Format, Scheme, Trim
from matplotlib.pyplot import title, xlabel
from numpy.lib.function_base import select
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
#import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os, sys
from collections import OrderedDict
sys.path.append("../..")
sys.path.append(r"C:/Users/joemo/Code/Repos/physrisk/src")

from dash.dependencies import Output, Input
import physrisk
from physrisk.kernel import PowerGeneratingAsset
from physrisk.kernel import calculate_impacts

cache_folder = r"C:/Users/joemo/Code/Repos/WRI-EBRD-Flood-Module/data_1"
asset_list = pd.read_csv(os.path.join(cache_folder, "wri-all.csv"))

types = asset_list["primary_fuel"].unique()

asset_filt = asset_list.loc[asset_list['primary_fuel'] == 'Gas'] # Nuclear

defaultIndex = np.flatnonzero(asset_filt['gppd_idnr'] == 'GBR1000313')[0]  #WRI1023786 #WRI1023786

ids = np.array(asset_filt['gppd_idnr'])
names = np.array(asset_filt['name'])
longitudes = np.array(asset_filt['longitude'])
latitudes = np.array(asset_filt['latitude'])
generation = np.array(asset_filt['estimated_generation_gwh'])

all_assets = [PowerGeneratingAsset(lat, lon, generation = gen, primary_fuel = 'gas') for lon, lat, gen in zip(longitudes, latitudes, generation)]

def create_map_fig(assets):
    lats = [a.latitude for a in assets]
    lons = [a.longitude for a in assets]

    color_discrete_sequence = px.colors.qualitative.Pastel1
    colors_map = px.colors.qualitative.T10
    magnitudes = np.ones(len(lats))
    map_fig = go.Figure(
        go.Scattermapbox(
            lat=lats, 
            lon=lons, 
            hovertext=names,
            hoverinfo='text',
            marker=go.scattermapbox.Marker(
                size = 15,
                color = colors_map[0],
                allowoverlap = False,
                #line=dict(width=2, color='DarkSlateGrey')
            ),            
            selected=go.scattermapbox.Selected(
                marker = go.scattermapbox.selected.Marker(
                    size = 20,
                    color = colors_map[2])),
            selectedpoints = [defaultIndex]
        ),
        layout = go.Layout(
            mapbox_style="stamen-terrain", # "carto-positron", # "stamen-terrain",
            mapbox_center_lon=0,
            mapbox_center_lat=10,
            margin={"r":0,"t":0,"l":0,"b":0},
            clickmode='event+select'))

    return map_fig

def get_fig_for_asset(asset):
        
    fig = make_subplots(rows=1, cols=2)
    detailed_results = calculate_impacts([asset], cache_folder = cache_folder)  
    go1, go2 = get_fig_gos_for_asset(asset, detailed_results)

    fig.add_trace(
        go1,
        row=1, col=1
    )

    fig.add_trace(
        go2,
        row=1, col=2
    )
    
    fig.update_traces(marker_color = px.colors.qualitative.T10[0]) #'rgb(158,202,225)')
    fig.update_xaxes(title = 'Impact (generation loss in days per year)', row=1, col=1)
    fig.update_yaxes(title = 'Probability', tickformat = ',.2%', row=1, col=1),
    fig.update_xaxes(title = 'Exceedance probability', row=1, col=2)
    fig.update_xaxes(autorange="reversed", row=1, col=2)
    fig.update_yaxes(title = 'Loss (EUR)', row=1, col=2)
    fig.update_layout(showlegend = False, font_family='Lato, sans-serif',
        title='Average annual loss of {:.1f} generation days'.format(detailed_results[asset].impact.mean_impact()),
        title_font_size = 24, margin=dict(l=100, r=100, t=100, b=100))

    return fig

def get_fig_gos_for_asset(asset, detailed_results):
  
    res = detailed_results[asset]

    color_discrete_sequence = px.colors.qualitative.Pastel1

    impact_bins = res.impact.impact_bins
    impact_bin_probs = res.impact.prob

    go1 = go.Bar(x = 0.5*(impact_bins[0:-1] + impact_bins[1:]), y = impact_bin_probs, width = impact_bins[1:] - impact_bins[:-1])

    exc = res.impact.to_exceedance_curve()
    go2 = go.Scatter(x=exc.probs, y=exc.values * asset.generation * 100 * 1000)

    return go1, go2


asset_categories = np.array(["Power generating assets"])
hazards = np.array(["Inundation"])
date_start = datetime(2080, 1, 1)
date_end = datetime(2080, 1, 1)

map_fig = create_map_fig(all_assets)

external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
        "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = "Asset loss drill-down"

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.P(children="OS-C", className="header-subtitle"),
                html.H1(
                    children="Asset loss drill-down", className="header-title"
                ),
                html.P(
                    children="Drill into financial loss"
                    " at asset level, including asset hazard event"
                    " and vulnerability",
                    className="header-description",
                ),
            ],
            style = { 'background-image' :  'url("/assets/banner3.jpg', 
                'background-position' : 'center',
                'background-repeat' : 'no-repeat',
                'background-size' : '100%' },
            className="header",
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Asset Category", className="menu-title"),
                        dcc.Dropdown(
                            id="asset-filter",
                            options=[
                                {"label": asset, "value": asset}
                                for asset in np.sort(asset_categories)
                            ],
                            value=asset_categories[0],
                            clearable=False,
                            className="dropdown",
                        ),
                    ],
                    style= {'width': '20%', 'height': '100%' },
                ),
                html.Div(
                    children=[
                        html.Div(children="Hazard", className="menu-title"),
                        dcc.Dropdown(
                            id="hazard-filter",
                            options=[
                                {"label": hazard, "value": hazard}
                                for hazard in np.sort(hazards)
                            ],
                            value="Inundation",
                            clearable=False,
                            searchable=False,
                            className="dropdown",
                        ),
                    ],
                    style= {'width': '20%', 'height': '100%'  },
                ),
                html.Div(
                    children=[
                        html.Div(
                            children="Date Range", className="menu-title"
                        ),
                        dcc.DatePickerRange(
                            id="date-range",
                            min_date_allowed=date_start,
                            max_date_allowed=date_end,
                            start_date=date_start,
                            end_date=date_end,
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        html.Div(children="Scenarios", className="menu-title"),
                        dcc.Dropdown(
                            id="scenario-filter",
                            options=[
                                {"label": scenario, "value": scenario}
                                for scenario in ["base", "4.5", "8.5"]
                            ],
                            multi=True,
                            value="4.5",
                            clearable=False,
                            searchable=False,
                            className="dropdown",
                        ),
                    ],
                    style= {'width': '20%', 'height': '100%'  },
                ),
            ],
            className="menu",
        ),
        html.Div(
            children=[
                html.Div(
                    children=html.H1(id="asset-category-header", children=asset_categories[0]
                    ),
                ),
                html.Div(
                    children=dcc.Graph(
                        id="map-chart",
                        #config={"displayModeBar": False},
                        figure=map_fig
                    ),
                    className="card",
                ),
                html.Div(
                    children=html.H1(id="asset-header", children="Asset name"
                    ),
                ),
                html.Div(
                    children=dash_table.DataTable(
                        id='asset-data-table',
                        data = asset_filt.iloc[defaultIndex:defaultIndex + 1].to_dict('records'),
                        columns = [
                            dict(id='gppd_idnr', name='ID'),
                            dict(id='name', name='Name'),
                            dict(id='primary_fuel', name='Primary Fuel'),
                            dict(id='estimated_generation_gwh', name='Annual Generation (GWh)', type='numeric', format=Format(precision=1, scheme=Scheme.fixed)), 
                        ],
                        style_cell={'textAlign': 'center', 'font-family': "Lato, sans-serif"},
                        style_header={ 'fontWeight': 'bold' },
                        style_as_list_view=True
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="exceedance-chart",
                        config={"displayModeBar": False}
                    ), 
                    className="card",   
                )],
            className="wrapper",
        ),
    ]
)

@app.callback(
    [Output('asset-header', 'children'), Output('asset-data-table', 'data'), Output('exceedance-chart', 'figure')],
    [Input('map-chart', 'clickData')])
def display_click_data(clickData):
    #columns=[{"name": i, "id": i} for i in df.columns],
                        #data=df.to_dict('records'),

    index = defaultIndex if clickData is None else clickData['points'][0]['pointIndex']
    data = asset_filt.iloc[index:index + 1].to_dict('records')

    fig = get_fig_for_asset(all_assets[index])

    return names[index], data, fig #json.dumps(clickData, indent=2)

"""
@app.callback(
    Output("exceedance-chart", "figure")
    [
        Input("asset-filter", "value"),
        Input("hazard-filter", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
    ],
)
def update_charts(asset, hazard, start_date, end_date):
    from plotly.subplots import make_subplots
    
    fig = get_fig_for_asset(all_assets[index])

    return fig
"""

if __name__ == "__main__":
    app.run_server(debug=True)

