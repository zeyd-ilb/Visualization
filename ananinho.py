import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from dash import dcc, html
import plotly.express as px
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN

# Load the shark attack data from a CSV file
data = pd.read_csv('Australian Shark-Incident Database Public Version.csv')

# Example structure of CSV:
# Latitude, Longitude, Reference, Description

# Load a GeoJSON file with only Australia boundaries
geojson_url = "https://raw.githubusercontent.com/rowanhogan/australian-states/master/states.geojson"
geojson_data = requests.get(geojson_url).json()

# Generate fake data to simulate a gradient
# Use the longitude of each state to generate a gradient effect
state_data = pd.DataFrame({
    "STATE_NAME": ["New South Wales", "Victoria", "Queensland", "Western Australia", "South Australia", "Tasmania", "Northern Territory", "Australian Capital Territory"],
    "gradient": np.linspace(0.5, 1, 8),  # Gradient from left (0) to right (1)
})

# Create a gradient-filled map of Australia
fig = px.choropleth_mapbox(
    state_data,
    geojson=geojson_data,
    featureidkey="properties.STATE_NAME",  # Matches GeoJSON state names
    locations="STATE_NAME",  # Column in state_data
    color="gradient",  # Numerical column for gradient
    color_continuous_scale="Blues",  # Gradient scale: Blues
    mapbox_style="white-bg",  # No other lands visible
)

# Find all unique shaark common names and choosse a color palette
shark_types = data["Shark.common.name"].unique()
color_palette = sns.color_palette("Set1", len(shark_types))  # Use seaborn color palette\
shark_color_map = {shark_types[i]: mcolors.to_hex(color_palette[i]) for i in range(len(shark_types))}


location_type = data["Location"].unique()
color_palette_locaiton = sns.color_palette("Set1", len(location_type)) 
location_color_map = {location_type[i]: mcolors.to_hex(color_palette_locaiton[i]) for i in range(len(location_type))}

data["Location_color"] = data["Location"].map(location_color_map)

# Apply the color map to the data
data["Shark.color"] = data["Shark.common.name"].map(shark_color_map)

counts = data.groupby(['Location']).size().reset_index(name='Counts')

###################################################################################################################################################################
data["Latitude"] = [float(x) for x in data["Latitude"]]
data["Longitude"] = [float(x) for x in data["Longitude"]]

coords = data[["Latitude", "Longitude"]]
kms_per_radian = 6371.0088
epsilon = 130 / kms_per_radian
db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
cluster_labels = db.labels_
data["Cluster"] = cluster_labels
cluster_medians = data.groupby("Cluster")[["Latitude", "Longitude"]].median()

# MAKE GAPH ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add points for shark attack incidents
fig.add_scattermapbox(
    lat=cluster_medians['Latitude'],
    lon=cluster_medians['Longitude'],
    mode="markers",
    marker=dict(size=15, color='blue'),  # Small red markers for incidents
    text="Gold Coast",
    hoverinfo="text",  # Show text in hover tooltip
    selected=go.scattermapbox.Selected(marker = {"color":"red", "size":25})
)

# Explicitly re-apply the center and zoom settings to keep the map focused on Australia
fig.update_layout(
    coloraxis_showscale=False,
    mapbox=dict(
        center={"lat": -25.0, "lon": 134.0},  # Center on Australia
        zoom=3.5,  # Adjust zoom level to tightly frame Australia
    ),
    margin={"l": 0, "r": 0, "t": 0, "b": 0},  # Remove margins
)

# Dash app layout
app = dash.Dash(__name__)

app.layout = html.Div([

    html.Div([
        html.Label("Select Attribute for Bar Chart:", style={'marginBottom': '0.5%', 'marginTop': '2%'}),
        dcc.Dropdown(
            id='dropdown-axis-bar',
            options=[
                {'label': 'Injury Severity', 'value': 'Injury.severity'},
                {'label': 'Shark Species', 'value': 'Shark.common.name'},
                {'label': 'Location', 'value': 'Site.category'},
                {'label': 'Victim Activity', 'value': 'Victim.activity'}
            ],
            value='Injury.severity',  # Default value
            clearable=False,
            style={'marginBottom': '2%', 'marginTop': '2%'}
        ),
    ], style={"position": "fixed", 
              "z-index": "10",
              'width': '45%', 
              "right": "2%", 
              'margin': 'auto', 
              }),
    

    html.Div([
        dcc.Graph(
            id = 'map_points',
            figure=fig,
            style={"position": "absolute", "height": "100%", "width": "100%"},
            #hoverData={'points': [{'customdata': 'Australia'}]}
        ),
    ],

    style={
    "position": "fixed",  # Fix the div to the top-left corner
    "top": "10%",           # Position at the very top
    "left": "2%",          # Position at the very left
    "margin": "0",
    "padding": "0",
    "height": "60%",    # Match the height of the graph
    "width": "45%",     # Match the width of the graph
    "background-color": "#FF5733",  # Optional background color
    "box-shadow": "0 2px 5px rgba(0,0,0,0.2)"  # Optional shadow for styling
    }
    ),

    html.Div([
        dcc.Graph(
            id='bar-chart', 
            style={'margin': 'auto'})
    ],
    style={"position": "fixed", 
              'width': '50%', 
              "right": "2%", 
              'margin': 'auto', 
              }),

     html.Div([
        html.Label("Select X-Axis for Scatter Plot:", style={'marginBottom': '10px'}),
        dcc.Dropdown(
            id='dropdown-axis-scatter',
            options=[
                {'label': 'Distance from Shore', 'value': 'Distance.to.shore.m'},
                {'label': 'Shark Species', 'value': 'Shark.common.name'},
                {'label': 'Water Depth (m)', 'value': 'Depth.of.incident.m'},
                {'label': 'Air Temperature (°C)', 'value': 'Air.temperature.°C'}
            ],
            value='Distance.to.shore.m',  # Default value
            clearable=False,
            style={'marginBottom': '20px'}
        ),
        dcc.Graph(id='scatterplot', style={'margin': 'auto'})
    ], style={"position": "fixed", "right": "2%", 'width': '50%', 'margin': 'auto', 'marginTop': '20%'})
])

@app.callback(
    Output('bar-chart', 'figure'),
    [Input('dropdown-axis-bar', 'value')]
)
def update_bar_chart(selected_attribute):
    # Count occurrences of each unique value in the selected attribute
    counts = data[selected_attribute].value_counts().reset_index()
    counts.columns = [selected_attribute, 'Occurrences']

    # Create a bar chart using Plotly Express
    figs = px.bar(
        counts,
        x=selected_attribute,
        y='Occurrences',
        title=f"{selected_attribute} by Frequency",
        labels={selected_attribute: selected_attribute, 'Occurrences': 'Number of Occurrences'},
        template='plotly',
        color=selected_attribute  # Color bars based on the selected attribute
    )

    # Adjust layout for better readability
    figs.update_layout(xaxis={'categoryorder': 'total descending'})
    return figs

@app.callback(
    Output('scatterplot', 'figure'),
    [Input('dropdown-axis-scatter', 'value')]
)
def update_scatter_plot(selected_x_axis):
    # Create a scatter plot with selected X-axis
    fig = px.scatter(
        data, 
        x=selected_x_axis, 
        y='Injury.severity', 
        color='Injury.severity',
        title=f"Shark Attack Severity vs {selected_x_axis}",
        labels={
            'Injury.severity': 'Injury Severity',
            selected_x_axis: selected_x_axis.replace('.', ' ').title()  # Format axis label nicely
        },
        category_orders={'Injury.severity': ['minor', 'major', 'fatal']}  # Ensure correct order
    )

    # Update layout for better readability
    fig.update_layout(title=f"Shark Attack Severity vs {selected_x_axis}", template="plotly")
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)

# -*- coding: utf-8 -*-

import os
import dash
from dash import dcc
from dash import html
import plotly.express as px
import plotly.graph_objects as go

from dash import Input, Output, State


# main entry
app = dash.Dash(__name__)
app.title = 'test app'

mapbox_token =''

fig = go.Figure( )
fig.add_trace( go.Scattermapbox(
    name='deposits_all',
    lat=[ 0 ],
    lon=[ 0 ],
    mode='markers',
    marker=go.scattermapbox.Marker(
        size=20, color='blue',
        ),
    )
    )
fig.update_layout(
    mapbox = {
        'accesstoken':mapbox_token,
        'style': "mapbox://styles/mapbox/dark-v10",
        'center': {'lon': 0, 'lat': 0 },
        'zoom': 4,
        },
    margin={"r":0,"t":0,"l":0,"b":0},
    showlegend = False,
    )


@app.callback(
    Output( 'graph_map', 'figure' ),
    inputs=dict( 
        input=Input( 'button', 'n_clicks' ), 
        fig_map=State( 'graph_map', 'figure' ),
        ),
    prevent_initial_call=True,
)
def update_loc( input, fig_map  ):
    # print( fig_map['data'] )

    # fig_map['data'][0].update( dict(
    #     lat=[ fig_map['data'][-1]['lat'][0]+1 ],
    #     lon=[ fig_map['data'][-1]['lon'][0]+1 ],
    #     marker=dict(size=20, color='blue')
    #     )
    # )

    # return fig_map

    print( fig_map['data'] )
    fig = go.Figure( )
    fig.add_trace( 
        go.Scattermapbox(
            name='deposits_all',
            lat=[ input ],
            lon=[ input ],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=20, color='blue',
                ),
            )
        )
    fig.update_layout(
        mapbox = {
            'accesstoken':mapbox_token,
            'style': "mapbox://styles/mapbox/dark-v10",
            'center': {'lon': 0, 'lat': 0 },
            'zoom': 4,
            },
        # height=600,
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend = False,
        )

    # fig = px.scatter_mapbox( lat=[ fig_map['data'][-1]['lat'][0]+1 ], lon=[ fig_map['data'][-1]['lat'][0]+1 ] )

    return fig



app.layout = html.Div( 
        [
            html.Button( 'Add', id='button' ),
            dcc.Graph( figure=fig, id='graph_map', style={'height':'500px'} ),
        ] )

if __name__ == '__main__':
    app.run_server(
        debug=True,
        dev_tools_hot_reload = True
        )

# -*- coding: utf-8 -*-

import os
import dash
from dash import dcc
from dash import html
import plotly.express as px
import plotly.graph_objects as go

from dash import Input, Output, State


# main entry
app = dash.Dash(__name__)
app.title = 'test app'

mapbox_token =''

fig = go.Figure( )
fig.add_trace( go.Scattermapbox(
    name='deposits_all',
    lat=[ 0 ],
    lon=[ 0 ],
    mode='markers',
    marker=go.scattermapbox.Marker(
        size=20, color='blue',
        ),
    )
    )
fig.update_layout(
    mapbox = {
        'accesstoken':mapbox_token,
        'style': "mapbox://styles/mapbox/dark-v10",
        'center': {'lon': 0, 'lat': 0 },
        'zoom': 4,
        },
    margin={"r":0,"t":0,"l":0,"b":0},
    showlegend = False,
    )


@app.callback(
    Output( 'graph_map', 'figure' ),
    inputs=dict( 
        input=Input( 'button', 'n_clicks' ), 
        fig_map=State( 'graph_map', 'figure' ),
        ),
    prevent_initial_call=True,
)
def update_loc( input, fig_map  ):
    # print( fig_map['data'] )

    # fig_map['data'][0].update( dict(
    #     lat=[ fig_map['data'][-1]['lat'][0]+1 ],
    #     lon=[ fig_map['data'][-1]['lon'][0]+1 ],
    #     marker=dict(size=20, color='blue')
    #     )
    # )

    # return fig_map

    print( fig_map['data'] )
    fig = go.Figure( )
    fig.add_trace( 
        go.Scattermapbox(
            name='deposits_all',
            lat=[ input ],
            lon=[ input ],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=20, color='blue',
                ),
            )
        )
    fig.update_layout(
        mapbox = {
            'accesstoken':mapbox_token,
            'style': "mapbox://styles/mapbox/dark-v10",
            'center': {'lon': 0, 'lat': 0 },
            'zoom': 4,
            },
        # height=600,
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend = False,
        )

    # fig = px.scatter_mapbox( lat=[ fig_map['data'][-1]['lat'][0]+1 ], lon=[ fig_map['data'][-1]['lat'][0]+1 ] )

    return fig

    

    

app.layout = html.Div( 
        [
            html.Button( 'Add', id='button' ),
            dcc.Graph( figure=fig, id='graph_map', style={'height':'500px'} ),
        ] )

if __name__ == '__main__':
    app.run_server(
        debug=True,
        dev_tools_hot_reload = True
        )