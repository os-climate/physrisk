"""
Use dash to produce a visualization for an asset-level drill-down
Dash app then migrated to React / Flask
"""

from datetime import datetime
from typing import List
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
import os, sys, json
from collections import OrderedDict
import dash_leaflet as dl
import dash_leaflet.express as dlx
import dash_bootstrap_components as dbc


sys.path.append("../..")

from dash.dependencies import Output, Input, State
import physrisk
from physrisk.kernel.calculation import DetailedResultItem
from physrisk.kernel import PowerGeneratingAsset, Asset
from physrisk.kernel import calculate_impacts, curve

cache_folder = r"<cache folder>"
with open(os.path.join(cache_folder, 'colormap.json')) as jf:
    color_map_info = json.load(jf)

asset_list = pd.read_csv(os.path.join(cache_folder, "wri-all.csv"))

types = asset_list["primary_fuel"].unique()

asset_filt = asset_list.loc[asset_list['primary_fuel'] == 'Gas'] # Nuclear

asset_filt = asset_filt.append({'gppd_idnr' : 'WRI12541', 'name' : 'Turkana 1', 'primary_fuel' : 'Gas', 'longitude' : 36.596, 'latitude' : 2.898, 'capacity_mw' : 102, 'estimated_generation_gwh' : 394.4 }, ignore_index=True)

defaultIndex = np.flatnonzero(asset_filt['gppd_idnr'] == 'WRI12541')[0] #GBR1000313 #WRI1023786 #WRI1023786 #WRI1006025

ids = np.array(asset_filt['gppd_idnr'])
names = np.array(asset_filt['name'])
longitudes = np.array(asset_filt['longitude'])
latitudes = np.array(asset_filt['latitude'])
generation = np.array(asset_filt['estimated_generation_gwh'])
capacity = np.array(asset_filt['capacity_mw'])

all_assets = [PowerGeneratingAsset(lat, lon, generation = gen, capacity = cap, primary_fuel = 'gas', name = name, id = id) for lon, lat, gen, cap, name, id in zip(longitudes, latitudes, generation, capacity, names, ids)]

detailed_results = calculate_impacts(all_assets[defaultIndex:defaultIndex+1], cache_folder = cache_folder) 
impact_bins = detailed_results[all_assets[defaultIndex]].impact.impact_bins
impact_bins = curve.process_bin_edges_for_graph(impact_bins)


impacts_file = os.path.join(cache_folder, 'all_impacts.json')
if os.path.isfile(impacts_file):
    with open(impacts_file, 'r') as r:
        mean_loss = json.load(r)

else:
    full_detailed_results = calculate_impacts(all_assets, cache_folder = cache_folder) 
    mean_loss = {a.id : full_detailed_results[a].impact.mean_impact() for a in all_assets}
    with open(os.path.join(cache_folder, 'all_impacts.json'), 'w') as w:
        contents = json.dumps(mean_loss)
        w.write(contents)

def create_map_fig_leaflet(assets):
    markers = []
    for asset in assets:
        markers.append(dict(
        title = asset.name, 
        name = asset.name,
        primary_fuel = asset.primary_fuel,
        tooltip="Asset: <b>" + asset.name + '</b><br>Type: <b>' + asset.primary_fuel +"</b>",
        id = asset.id,
        lat = asset.latitude, 
        lon = asset.longitude))
        
    access_token = '<access token>'

    wri_tile =dl.Overlay(children=[dl.TileLayer(
        url = 'https://api.mapbox.com/v4/joemoorhouse.32lvye13/{z}/{x}/{y}@2x.png?access_token=' + access_token,
        attribution = "Riverine inundation")], name="Riverine inundation")

    cl = dl.Overlay(children=[dl.GeoJSON(data=dlx.dicts_to_geojson(markers), cluster=True, id='markers')], 
        id='clusters', name="Power generating assets (gas)")

    map = dl.Map( 
        dl.LayersControl(
            [ dl.BaseLayer(dl.TileLayer(), name="Base Layer"), wri_tile, cl]
        ), id='the-map', preferCanvas=True, center=[39, -98], zoom=4       
    )

    return (map)


def create_map_fig(assets, layers):
    lats = [a.latitude for a in assets]
    lons = [a.longitude for a in assets]

    color_discrete_sequence = px.colors.qualitative.Pastel1
    colors_map = px.colors.qualitative.T10
    magnitudes = [mean_loss[a.id] for a in assets] 

    map_fig = go.Figure(
        go.Scattermapbox(   
            lat=lats, 
            lon=lons, 
            hovertext=names,
            hoverinfo='text',
            marker=go.scattermapbox.Marker(
                size = 15,
                color = magnitudes,
                #color = colors_map[0],
                allowoverlap = False,
                colorscale = 'Aggrnyl',
            ),            
            selected=go.scattermapbox.Selected(
                marker = go.scattermapbox.selected.Marker(
                    size = 20,
                    color = colors_map[2])),
            #selectedpoints = [defaultIndex]
        ),

        layout = go.Layout(
            mapbox_style="stamen-terrain", 
            mapbox_center_lon=0,
            mapbox_center_lat=10,
            margin={"r":0,"t":0,"l":0,"b":0},
            clickmode='event+select'
        ))
    access_token = '<api token here>'
    map_fig.layout.mapbox.accesstoken = access_token
    url = 'https://api.mapbox.com/v4/joemoorhouse.0zy9pvov/{z}/{x}/{y}@2x.png?access_token=' + access_token
    map_fig.layout.mapbox.layers = [
                    {
                        "below":"traces",
                        "sourcetype": "raster",
                        "source" : [url],
                        "sourceattribution": "WRI",
                        "visible": True
                    }]
    
    update_map_fig_layers(map_fig, layers)

    map_fig_colorbar = make_subplots(rows=1, cols=1)
    min_val, min_index = color_map_info['min']['data'], color_map_info['min']['color_index']
    max_val, max_index = color_map_info['max']['data'], color_map_info['max']['color_index']
    colorscale = []
    for i in range(min_index, max_index + 1):
        (r, g, b, a) = color_map_info['colormap'][str(i)]
        colorscale.append([(float(i) - min_index) / (max_index - min_index) , f'rgb({r}, {g}, {b})'])

    levels = np.linspace(0.0, 2.0, 256)

    colorbar = go.Heatmap(
                  x = levels,
                  y = [0.0, 1.0],
                  z = [levels[:-1]],
                  colorscale = colorscale, #'Reds',
                  hoverongaps = False,
                  hovertemplate='')

    map_fig_colorbar.add_trace(
       colorbar,
       row=1, col=1
    )

    map_fig_colorbar.update_layout(height=60, margin=dict(l=20, r=20, t=35, b=5), showlegend = False, font_family='Lato, sans-serif', font_size = 16, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    map_fig_colorbar.update_xaxes(title_font = {"size": 20}, ticksuffix = 'm') #title = 'Intensity' 
    map_fig_colorbar.update_yaxes(visible = False, showticklabels = False, title = 'Inundation depth')
    map_fig_colorbar.update_traces(showscale = False) #'rgb(158,202,225)')
    return map_fig, map_fig_colorbar

def update_map_fig_layers(map_fig, layers):
    with map_fig.batch_update():
        map_fig.data[0].marker.size = 15 if "A" in layers else 0
        map_fig.data[0].selected.marker.size = 15 if "A" in layers else 0
        map_fig.layout.mapbox.layers[0]["visible"] = True if "I1000" in layers else False
    return

def get_fig_for_asset(asset, detailed_results):
        
    fig = make_subplots(rows=1, cols=2) 
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
    fig.update_xaxes(title = 'Impact (generation loss in days per year)', title_font = {"size": 20}, row=1, col=1)
    fig.update_yaxes(title = 'Probability', title_font = {"size": 20}, ticksuffix = '%', row=1, col=1),
    fig.update_xaxes(title = 'Exceedance probability', title_font = {"size": 20}, type="log", row=1, col=2)
    fig.update_xaxes(autorange="reversed", ticksuffix = '%', row=1, col=2) 
    fig.update_yaxes(title = 'Loss (EUR)', title_font = {"size": 20}, row=1, col=2)
    
    headline = 'Average annual loss of {:.1f} generation days'.format(detailed_results[asset].impact.mean_impact())
    curve = detailed_results[asset].impact.to_exceedance_curve()
    for p in [0.1, 0.05, 0.01, 0.005, 0.001]:
        if p >= curve.probs[-1] and p <= curve.probs[0]:
            loss = curve.get_value(p)
            headline = '{0:.1g}% probability of annual loss greater than {1:.1f} generation days'.format(p * 100, loss)
            break

    fig.update_layout(showlegend = False, font_family='Lato, sans-serif', font_size = 16,
        title=headline,
        title_font_size = 24, margin=dict(l=100, r=100, t=100, b=100))

    return fig

def get_fig_gos_for_asset(asset, detailed_results):
  
    res = detailed_results[asset]

    color_discrete_sequence = px.colors.qualitative.Pastel1

    impact_bins = res.impact.impact_bins
    impact_bin_probs = res.impact.prob

    impact_bins = curve.process_bin_edges_for_graph(impact_bins)

    with open(os.path.join(cache_folder, 'log.txt'), 'w') as f:
        contents = json.dumps(str(impact_bins))
        f.write(contents)

    go1 = go.Bar(x = 0.5*(impact_bins[0:-1] + impact_bins[1:]), y = impact_bin_probs * 100, width = impact_bins[1:] - impact_bins[:-1])

    exc = res.impact.to_exceedance_curve()
    go2 = go.Scatter(x=exc.probs * 100, y=exc.values * asset.generation * 100 * 1000 / 365)

    return go1, go2

def get_fig_for_model(asset, detailed_results):
        
    fig = make_subplots(rows=1, cols=2)
    go1, go2, headline = get_fig_gos_for_model(asset, detailed_results)

    fig.add_trace(
        go1,
        row=1, col=1
    )

    fig.add_trace(
        go2,
        row=1, col=2
    )
    
    fig.update_traces(marker_color = px.colors.qualitative.T10[0]) #'rgb(158,202,225)')
    fig.update_xaxes(title = 'Inundation intensity (m)', title_font = {"size": 20}, row=1, col=1)
    fig.update_yaxes(title = 'Probability', title_font = {"size": 20}, ticksuffix = "%", row=1, col=1),
    fig.update_xaxes(title = 'Exceedance probability', title_font = {"size": 20}, type="log", row=1, col=2)
    fig.update_xaxes(autorange="reversed", row=1, col=2, ticksuffix = "%")
    fig.update_yaxes(title = 'Inundation intensity (m)', title_font = {"size": 20}, row=1, col=2) 
    fig.update_layout(showlegend = False, font_family='Lato, sans-serif', font_size = 16,
        title=headline, # 'Inundation intensity',
        title_font_size = 24, margin=dict(l=100, r=100, t=100, b=100))

    return fig

def get_fig_gos_for_model(asset, detailed_results):
  
    res : DetailedResultItem = detailed_results[asset]

    color_discrete_sequence = px.colors.qualitative.Pastel1

    intensity_bins = res.event.intensity_bins
    intensity_bin_probs = res.event.prob

    go1 = go.Bar(x = 0.5*(intensity_bins[0:-1] + intensity_bins[1:]), y = intensity_bin_probs * 100, width = intensity_bins[1:] - intensity_bins[:-1])

    exc = res.event.exceedance # to_exceedance_curve()
    go2 = go.Scatter(x=exc.probs * 100, y=exc.values)

    headline = '10% probability of event with intensity greater than {0:.2f}m in a single year'.format(exc.get_value(0.1))

    return go1, go2, headline

def get_fig_for_vulnerability(asset, detailed_results):
        
    fig = make_subplots(rows=1, cols=1)
    go1 = get_fig_gos_for_vulnerability(asset, detailed_results)

    fig.add_trace(
        go1,
        row=1, col=1
    )
    
    fig.update_xaxes(title = 'Inundation intensity (m)', title_font = {"size": 20}, row=1, col=1)
    fig.update_yaxes(title = 'Impact (generation loss)', title_font = {"size": 20}, row=1, col=1),
    fig.update_layout(showlegend = False, font_family='Lato, sans-serif', font_size = 16,
        title='Vulnerability distribution',
        title_font_size = 24, margin=dict(l=100, r=100, t=100, b=100))

    return fig

def get_fig_gos_for_vulnerability(asset, detailed_results):
  
    res : DetailedResultItem = detailed_results[asset]

    go1=go.Heatmap(
                   z=res.vulnerability.prob_matrix,
                   x=res.vulnerability.intensity_bins,
                   y=res.vulnerability.impact_bins,
                   colorscale = 'Reds',
                   hoverongaps = False,
                   hovertemplate='')
    return go1


asset_categories = np.array(["Power generating assets (gas)"])
hazards = np.array(["Inundation"])
date_start = datetime(2080, 1, 1)
date_end = datetime(2080, 12, 31)

#map_fig = create_map_fig_leaflet(all_assets)
map_fig, map_fig_colorbar = create_map_fig(all_assets, ["I1000"])

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
                html.P(children="OS-Climate", className="header-subtitle"),
                html.H1(
                    children="Asset loss drill-down", className="header-title"
                ),
                html.P(
                    children="Drill into financial loss"
                    " at asset level: loss, hazard intensity"
                    " and vulnerability",
                    className="header-description",
                ),
            ],
            style = { 'background-image' :  'url("/assets/banner3.jpg', 
                'background-position' : 'center',
                'background-repeat' : 'no-repeat',
                'background-size' : 2600 }, 
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
                    style= {'width': '25%'}#, 'height': '100%' },
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
                    style= {'width': '20%' }#, 'height': '100%'  },
                ),
                html.Div(
                    children=[
                        html.Div(
                            children="Dates", className="menu-title"
                        ),
                        dcc.Dropdown(
                            id="dates",
                            options=[
                                {"label": date, "value": date}
                                for date in ["Today", "2030", "2050", "2080"]
                            ],
                            multi=True,
                            value=["2080"],
                            clearable=False,
                            searchable=False,
                            className="dropdown",
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
                                for scenario in ["Histo", "RCP4.5", "RCP8.5"]
                            ],
                            multi=True,
                            value=["RCP8.5"],
                            clearable=False,
                            searchable=False,
                            className="dropdown",
                        ),
                    ],
                    #className="custom-dropdown",
                    #style= {'width': '15%' } #, 'height': '100%' },
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
                    dcc.Checklist(
                        options=[
                            {'label': 'Assets', 'value': 'A'},
                            {'label': 'Inundation 10-year', 'value': 'I10'},
                            {'label': 'Inundation 100-year', 'value': 'I100'},
                            {'label': 'Inundation 1000-year', 'value': 'I1000'}
                        ],
                        id='layer-select',
                        value=[],
                        labelStyle={'display': 'inline-block'}
                    ),
                    style = {'width': '100%', 'display': 'flex', 'align-items': 'right', 'justify-content': 'right'},
                ),
                html.Div(
                    children=[
                        dcc.Graph(
                            id="map-chart",
                            figure=map_fig,
                            style={'width': '100%', 'height': '40vh'}
                        ),
                    ],
                    style={'margin': '2px'},
                    className="card",
                ),
                html.Div([
                    html.Div(
                        [html.P(children="Inundation depth (m)")], 
                        #className="column",
                        style={'flex': '20%', 'margin': '0px 0px 0px 20px', 'font-size': 20, 'vertical-align': 'top'},
                    ),
                    html.Div(
                        children=dcc.Graph(
                            id="map-chart-colorbar",
                            config={"displayModeBar": False},
                            figure=map_fig_colorbar,
                        ),
                        style={'flex': '80%', 'padding': '0px 0px 0px 0px'},
                    ),
                    ],
                    id="map-chart-colorbar-container", 
                    style={'display':'flex'},
                    className="card"
                ),
                html.Div(id = 'drilldown', style = {'display' : 'none'}, className = "wrapper", children=[
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
                            dict(id='capacity_mw', name='Capacity (MW)'),
                            dict(id='estimated_generation_gwh', name='Est. annual Generation (GWh)', type='numeric', format=Format(precision=1, scheme=Scheme.fixed)), 
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
                ),
                html.Div(
                    children=dcc.Graph(
                        id="intensity-chart",
                        config={"displayModeBar": False}
                    ), 
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="vulnerability-chart",
                        config={"displayModeBar": False}
                    ), 
                    className="card",
                )
                ]),
                ],
            className="wrapper",
        ),
    ]
)

@app.callback(
    [Output('drilldown', 'style'), Output('asset-header', 'children'), Output('asset-data-table', 'data'), 
        Output('exceedance-chart', 'figure'), Output('intensity-chart', 'figure'), Output('vulnerability-chart', 'figure')],
    #[Input("markers", "click_feature")])
    [Input('map-chart', 'clickData')])
def display_click_data(clickData):
    visible = 'none' if clickData is None else 'block'
    #index = defaultIndex if click_feature is None else np.flatnonzero(asset_filt['gppd_idnr'] == click_feature["properties"]["id"])[0]
    index = defaultIndex if clickData is None else clickData['points'][0]['pointIndex']
    data = asset_filt.iloc[index:index + 1].to_dict('records')
    asset = all_assets[index]
    detailed_results = calculate_impacts([asset], cache_folder = cache_folder) 
    fig1 = get_fig_for_asset(asset, detailed_results)
    fig2 = get_fig_for_model(asset, detailed_results)
    fig3 = get_fig_for_vulnerability(asset, detailed_results)
    
    fig_plot = get_fig_for_model(asset, detailed_results)
    fig_plot.update_layout(width=900)
    fig_plot.write_image("C:/Users/joemo/Code/Repos/physrisk/docs/methodology/plots/fig_intensity.pdf")
    return {'display':visible }, names[index], data, fig1, fig2, fig3 #json.dumps(clickData, indent=2)

@app.callback(
    #[Output('drilldown', 'style'), 
    [Output('map-chart-colorbar-container', 'style'), Output('map-chart', 'figure')],#, Output('map-chart-colorbar', 'figure')],
    [Input('layer-select', 'value'), State('map-chart', 'figure')])
def update_map_layers(value, fig):

    lon = fig["layout"]["mapbox"]["center"]["lon"]
    lat = fig["layout"]["mapbox"]["center"]["lat"]
    zoom = fig["layout"]["mapbox"].get("zoom", 1.0)
    update_map_fig_layers(map_fig, value)
    display_colorbar = any(["I" in v for v in  value])
    display_asset = any(["A" in v for v in  value])
    map_fig.update_layout(mapbox_center_lon=lon,
            mapbox_center_lat=lat,
            mapbox_zoom=zoom)
    #{'display':'block'} if display_asset else {'display':'none'}, 
    return {'display':'flex'} if display_colorbar else {'display':'none'}, map_fig #, map_fig_colorbar

if __name__ == "__main__":
    app.run_server(debug=False)

