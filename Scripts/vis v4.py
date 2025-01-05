#interactive shark map - bar chart
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import requests
import plotly.express as px
import json
import random
from dash.dependencies import Input, Output, State
import dash_mantine_components as dmc
from sklearn.cluster import DBSCAN
import numpy as np




# Load the shark attack data
data = pd.read_csv('Australian Shark-Incident Database Public Version.csv')

# Load a GeoJSON file with only Australia boundaries
# geojson_url = "https://raw.githubusercontent.com/rowanhogan/australian-states/master/states.geojson"
# geojson_data = requests.get(geojson_url).json()


# Initial figure setup
initial_fig = go.Figure()

# Add shark incident points to the initial figure
initial_fig.add_trace(go.Scattermapbox(
    lat=data["Latitude"],
    lon=data["Longitude"],
    mode="markers",
    marker=dict(size=8, color="red"),
    text=data["Reference"],  # Hover text
    hoverinfo="text",  # Text displayed on hover
    customdata=data.index,  # Pass row indices as custom data
))

# Final map settings for the initial figure
initial_fig.update_layout(
    mapbox=dict(
        style="carto-darkmatter",
        zoom=3.5,
        center={"lat": -25.0, "lon": 134.0},  # Center the map on Australia
    ),
    margin={"l": 0, "r": 0, "t": 0, "b": 0}
)

# Create the base map
fig = go.Figure()

# Add a GeoJSON layer to show only Australia's boundaries
#TO NAME THE STATES
"""fig.add_trace(go.Choroplethmapbox(
    geojson=geojson_data,
    featureidkey="properties.STATE_NAME",
    locations=[],  # No data binding to states
    z=[],  # No color data
    showscale=False,  # Disable the color scale
))"""

data["Latitude"] = [float(x) for x in data["Latitude"]]
data["Longitude"] = [float(x) for x in data["Longitude"]]

coords = data[["Latitude", "Longitude"]]
kms_per_radian = 6371.0088
epsilon = 85/ kms_per_radian
db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
cluster_labels = db.labels_
data["Cluster"] = cluster_labels
outlisers = data.groupby("Cluster")[["Cluster","Latitude", "Longitude"]].count()
i = outlisers[(outlisers.Cluster < 4)].index
new_df = data.groupby("Cluster")[["Cluster","Latitude", "Longitude"]].median()
new_df.drop(i, inplace=True)

# Add shark incident points
fig.add_trace(go.Scattermapbox(
    lat=new_df["Latitude"],
    lon=new_df["Longitude"],
    mode="markers",
    marker=dict(size=8, color="red"),
    text=new_df["Location"],  # Hover text
    hoverinfo="text",  # Text displayed on hover
    customdata=data.index,  # Pass row indices as custom data
))

# Final map settings
fig.update_layout(
    mapbox=dict(
        # style="white-bg",  # Disable the base map (shows no other lands)
        style="carto-darkmatter",  # Disable the base map (shows no other lands)
        center={"lat": -25.0, "lon": 134.0},  # Center the map on Australia
        zoom=3.5,
        # layers=[
        #     dict(
        #         sourcetype="geojson",
        #         source=geojson_data,  # Use GeoJSON for Australia's boundaries
        #         below="traces",
        #         type="fill",
        #         color="lightblue",  # Fill Australia's boundaries with light blue
        #     )
        # ],
    ),
    margin={"l": 0, "r": 0, "t": 0, "b": 0},  # Remove margins
)

# Dash app
app = dash.Dash(__name__)

# App layout with a resizable sidebar
app.layout = dmc.MantineProvider(
    id="app",
    theme={"colorScheme": "dark"},
    children=[

        dmc.Grid(
            align="center",
            justify="center",
            gutter="0",
            style={"margin": 0, "padding": 0},  # Remove margin and padding from the grid

            children=[
                dmc.Col(
                    span=6,  # Adjust the span as needed
                    style={"padding": 0, "margin": 0, "height": "100vh", "flex": "1"},  # Sidebar resizes with the map
                    children=[
                        html.Div(
                            id="resizable-sidebar",
                            style={
                                "margin": 0,
                                "padding": "30px",
                                "height": "100vh",
                                "backgroundColor": "#000",  # Black background color
                                "color": "#fff",  # Light text color
                                "display": "flex",
                                "flexDirection": "column",
                                "overflow": "auto",  # Add scrollbars when needed
                                # "resize": "horizontal", # Not supported with MantineProvider
                            },
                            children=[
                                # Left COLUMN
                                html.H2("Shark Incident Details", style={"marginBottom": "10px", "color": "#fff"}),  # Light text color
                                html.Div("Click on a point to see details. (disabled for now)", id="incident-details", style={"marginBottom": "20px", "color": "#fff"}),  # Light text color
                                
                                #BAR CHART
                                html.Div(
                                    children=[
                                        html.Label("Select Attribute for Bar Chart:", style={'marginBottom': '10px', 'color': '#fff'}),  # Light text color
                                        dcc.Dropdown(
                                            id='dropdown-axis-bar',
                                            options=[
                                                {
                                                    "label": html.Span(['Injury Severity'], style={'color': '#fff', 'backgroundColor': '#000'}),  
                                                    "value": 'Injury.severity',
                                                },
                                                {
                                                    "label": html.Span(['Shark Species'], style={'color': '#fff', 'backgroundColor': '#000'}),
                                                    "value": 'Shark.common.name',
                                                },
                                                {
                                                    "label": html.Span(['Location'], style={'color': '#fff', 'backgroundColor': '#000'}),
                                                    "value": 'Location',
                                                },
                                                {
                                                    "label": html.Span(['Victim Activity'], style={'color': '#fff', 'backgroundColor': '#000'}),
                                                    "value": 'Victim.activity',
                                                }
                                            ],
                                            value='Injury.severity',  # Default value
                                            clearable=False,
                                            style={
                                                'marginBottom': '10px',
                                                'marginTop': '10px',
                                                'color': '#fff',  # Light text color
                                                'backgroundColor': '#000',  # Black background color
                                            }
                                        ),

                                        html.Label("Filter Options:", style={'marginBottom': '10px', "color": "#fff"}),  # Light text color
                                        dmc.SegmentedControl(
                                            id='filter-options-bar',
                                            style = {'marginBottom': '10px', "color": "#fff"},
                                            orientation="horizontal",
                                            fullWidth=True,
                                            value='top_10',  # Default option
                                            data=[
                                                {'label': '10 Most Frequent', 'value': 'top_10'},
                                                {'label': '10 Least Frequent', 'value': 'bottom_10'},
                                                {'label': 'Show All', 'value': 'all'}
                                                ]
                                        ),

                                        html.Label("Click on a bar to see the points:", style={'marginBottom': '10px', "color": "#fff"}), 
                                        dcc.Graph(id='bar-chart', style={'margin': '0'}),
                                        
                                ]),
                                html.Div(id='previous-dropdown-value', style={'display': 'none'}),
                                dmc.Switch(id="switch-example", label="Use Log Scale.", checked=False),
                                html.H2("", style={"marginBottom": "50px", "color": "#fff"}),  # To give unvisible bottom margin

                            ],
                        ),
                    ],
                ),
                dmc.Col(
                    span=6,  # Adjust the span as needed
                    style={"padding": 0, "margin": 0, "height": "100%", "width": "100%", "flex": "1", "overflow": "hidden"},  # Map resizes with the sidebar
                    children=[
                        dcc.Graph(
                            id="shark-map",
                            figure=fig,
                            style={"flex": "1", "overflow": "hidden","padding": 0, "margin": 0, "height":"100vh", "width":"100vh"},  # Map resizes with the sidebar
                        ),
                    ],
                ),
            ],
        ),
    ],
)

# Callback to update side panel on point click
"""{
@app.callback(
    # [Output("incident-details", "children"), Output("sidebar-bar-chart", "figure")],
    # [Output("incident-details", "children")],
    [Input("shark-map", "clickData")],
)
def display_incident_details_and_chart(clickData):
    if clickData is None:
        return "Click on a point to see details.", go.Figure()

    # Extract the index of the clicked point
    point_index = clickData["points"][0]["customdata"]

    # Retrieve details of the incident
    incident = data.iloc[point_index]

    # Format the incident details
    details = html.Div([
        html.P(f"Reference: {incident['Reference']}"),
        html.P(f"Date:  {incident['Incident.month']} / {incident['Incident.year']}"),
        html.P(f"Time: {incident['Time.of.incident']}")
    ])


    return details
}"""



# Callback to update the bar chart 
@app.callback(
    Output('bar-chart', 'figure'),
    [Input('dropdown-axis-bar', 'value'), Input('filter-options-bar', 'value'),Input("switch-example", "checked")]
)
def update_bar_chart(selected_attribute, filter_option, log_scale):
    # Count occurrences of each unique value in the selected attribute
    counts = data[selected_attribute].value_counts().reset_index()
    counts.columns = [selected_attribute, 'Occurrences']

    # Define a dictionary to map original column names to desired names
    column_name_mapping = {
        'Injury.severity': 'Injury Severity',
        'Shark.common.name': 'Shark Species',
        'Victim.activity': 'Victim Activity',
        # Add more mappings as needed
    }

    # Rename the columns in the DataFrame
    counts.rename(columns=column_name_mapping, inplace=True)

    # Update the selected_attribute if it is one of the renamed columns
    selected_attribute_renamed = column_name_mapping.get(selected_attribute, selected_attribute)

    # Apply filtering based on the selected filter option
    if filter_option == 'top_10':
        filtered_counts = counts.nlargest(10, 'Occurrences')  # Top 10
        colors = [f"rgb({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)})" for _ in range(len(filtered_counts))]
        category_order = 'total descending'
    elif filter_option == 'bottom_10':
        filtered_counts = counts.nsmallest(10, 'Occurrences')  # Bottom 10
        colors = [f"rgb({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)})" for _ in range(len(filtered_counts))]
        category_order = 'total ascending'
    else:  # Show all
        filtered_counts = counts
        colors = [f"rgb({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)})" for _ in range(len(filtered_counts))]
        category_order = 'total descending'

    # Create a bar chart using Plotly Graph Objects
    fig = go.Figure(data=[
        go.Bar(
            x=filtered_counts[selected_attribute_renamed],  # Set x-axis values
            y=filtered_counts['Occurrences'],  # Set y-axis values
            marker_color=colors,  # Set the colors of the bars (if needed)
            customdata=colors,  # Pass the colors as custom data
            hovertemplate='%{x}<br>Occurrences: %{y}<extra></extra>',  # Customize hover text
        )
    ])
   
    if log_scale:
        yaxis_type="log"
    else:
        yaxis_type="linear"
   
    # Adjust layout for better readability and additional options
    fig.update_layout(
        title=f"{selected_attribute_renamed} by Frequency ({filter_option.replace('_', ' ').capitalize()})",
        xaxis=dict(
            title= selected_attribute_renamed,
            categoryorder= category_order  # Sort categories by total descending
        ),
        yaxis=dict(
            title='Number of Occurrences',
            type= yaxis_type
        ),
        template='plotly_dark',  # Set the dark theme
    )

    
    return fig
    
'''
# Callback to update shark map based on bar click
@app.callback(
    [Output('shark-map', 'figure'),
     Output('previous-dropdown-value', 'children')],
    [Input('bar-chart', 'clickData'),
     Input('dropdown-axis-bar', 'value')],
    [State('previous-dropdown-value', 'children')]
)
def update_shark_map(clickData, selected_attribute, previous_attribute):
    if previous_attribute is None:
        previous_attribute = selected_attribute

    if clickData is None or selected_attribute != previous_attribute:
        return initial_fig, selected_attribute

    clicked_category = clickData['points'][0]['x']

    marker_color = clickData['points'][0]['customdata']  # Get the color of the clicked bar
    filtered_map_df = data[data[selected_attribute] == clicked_category]

    new_fig = go.Figure(go.Scattermapbox(
        lat=filtered_map_df['Latitude'],
        lon=filtered_map_df['Longitude'],
        mode='markers',
        marker=go.scattermapbox.Marker(size=8, color=marker_color),
        text=data['Reference'],
    ))
    new_fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            zoom=3.5,  
            center={"lat": -25.0, "lon": 134.0},  # Center the map on Australia
        ),
        margin={"l": 0, "r": 0, "t": 0, "b": 0}
    )
    return new_fig, selected_attribute
'''
    
if __name__ == "__main__":
    app.run_server(debug=True)