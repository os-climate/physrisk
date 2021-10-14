"""
Use dash to produce a visualization for an asset-level drill-down
Dash app then migrated to React / Flask
"""



from datetime import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
from numpy.lib.function_base import select
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os, sys
sys.path.append("../..")

from dash.dependencies import Output, Input
import physrisk
from physrisk.kernel import PowerGeneratingAsset
from physrisk.kernel import calculate_impacts

cache_folder = r"<cache folder>"
asset_list = pd.read_csv(os.path.join(cache_folder, "wri-all.csv"))

types = asset_list["primary_fuel"].unique()

asset_filt = asset_list.loc[asset_list['primary_fuel'] == 'Gas'] # Nuclear


ids = np.array(asset_filt['gppd_idnr'])
names = np.array(asset_filt['name'])
longitudes = np.array(asset_filt['longitude'])
latitudes = np.array(asset_filt['latitude'])
generation = np.array(asset_filt['estimated_generation_gwh'])

all_assets = [PowerGeneratingAsset(lat, lon, generation = gen, primary_fuel = 'gas') for lon, lat, gen in zip(longitudes, latitudes, generation)]

assets = all_assets[0:30]

detailed_results = calculate_impacts(assets, cache_folder = cache_folder)

def create_map_fig(assets):
    lats = [a.latitude for a in assets]
    lons = [a.longitude for a in assets]

    color_discrete_sequence = px.colors.qualitative.Pastel1
    magnitudes = np.ones(len(lats))
    map_fig = go.Figure(go.Scattermapbox(lat=lats, lon=lons, 
        marker=go.scattermapbox.Marker(size = 10, allowoverlap = False),
        selected=go.scattermapbox.Selected(marker = go.scattermapbox.selected.Marker(size = 10, color = color_discrete_sequence[1]))
        )) # z=magnitudes, adius=10) # Scattermapbox# Densitymapbox
                                 
    map_fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=0)
    map_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    return map_fig

def get_figs_for_asset(asset, detailed_results):
    #asset = list(detailed_results.keys())[0]
    res = detailed_results[asset]

    color_discrete_sequence = px.colors.qualitative.Pastel1

    impact_bins = res.impact.impact_bins
    impact_bin_probs = res.impact.prob

    fig1 = go.Figure(
        data = [go.Bar(x = 0.5*(impact_bins[0:-1] + impact_bins[1:]), y = impact_bin_probs, width = impact_bins[1:] - impact_bins[:-1])]
    )

    fig1.update_traces(marker_color = px.colors.qualitative.T10[0]) #'rgb(158,202,225)')
    fig1.update_xaxes(title = 'Impact (generation loss in days per year)')
    fig1.update_yaxes(title = 'Probability')

    exc = res.impact.to_exceedance_curve()
    fig2 = px.line(x=exc.probs, y=exc.values * asset.generation * 100 * 1000)
    fig2.update_xaxes(title = 'Exceedance probability')
    fig2.update_xaxes(autorange="reversed")
    fig2.update_yaxes(title = 'Loss (EUR)')

    return fig1, fig2

hazards = np.array(["Inundation"])
date_start = datetime(2080, 1, 1)
date_end = datetime(2080, 1, 1)

map_fig = create_map_fig(all_assets)
fig1, fig2 = get_figs_for_asset(assets[22], detailed_results)

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
                html.P(children="OS-C", className="header-emoji"),
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
            className="header",
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Asset", className="menu-title"),
                        dcc.Dropdown(
                            id="asset-filter",
                            options=[
                                {"label": asset, "value": asset}
                                for asset in np.sort(names)
                            ],
                            value=names[22],
                            clearable=False,
                            className="dropdown",
                        ),
                    ]
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
                    ]
                ),
            ],
            className="menu",
        ),
        html.Div(
            children=[
                html.Div(
                    children=dcc.Graph(
                        id="map-chart",
                        #config={"displayModeBar": False},
                        figure=map_fig
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="exceedance-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="loss-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
            ],
            className="wrapper",
        ),
    ]
)


@app.callback(
    [Output("exceedance-chart", "figure"), Output("loss-chart", "figure")],
    [
        Input("asset-filter", "value"),
        Input("hazard-filter", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
    ],
)
def update_charts(asset, hazard, start_date, end_date):

    #fig1, fig2 = get_figs_for_asset(assets[22], detailed_results)
    return fig2, fig1


if __name__ == "__main__":
    app.run_server(debug=True)

