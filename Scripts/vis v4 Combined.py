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
from dash import Input, Output, State
import seaborn as sns
import matplotlib.colors as mcolors


# Load the shark attack data
# data = pd.read_csv('Australian Shark-Incident Database Public Version.csv')
data = pd.read_csv("C:\\Users\\20223070\\Downloads\\sharks.csv")

# Load a GeoJSON file with only Australia boundaries
# geojson_url = "https://raw.githubusercontent.com/rowanhogan/australian-states/master/states.geojson"
# geojson_data = requests.get(geojson_url).json()

#Not being used now
#Initial figure with all data points
def create_initial_fig_all():
    # Initial figure setup
    initial_fig = go.Figure()

    # Add shark incident points to the initial figure
    initial_fig.add_trace(go.Scattermapbox(
        lat=data["Latitude"],
        lon=data["Longitude"],
        mode="markers",
        marker=dict(size=8, color="red"),
        text=data["Shark.common.name"],  # Hover text
        hoverinfo="text",  # Text displayed on hover
        customdata=data.index,  # Pass row indices as custom data
    ))

    # Final map settings for the initial figure
    initial_fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            zoom=3.5,
            center={"lat": -23.69748, "lon": 133.88362},  # Center the map on Australia
        ),
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        dragmode=False  # Disable dragging
    )
    return initial_fig

# initial_fig = create_initial_fig_all()

def location_cluster():
    # PREPARE DATA
    # CLUSTER LOCATIONS

    # Clean the Longitude column by removing trailing periods
    data["Longitude"] = data["Longitude"].str.rstrip('.')
    data["Longitude"] = [float(x) for x in data["Longitude"]]

    # Clean the Latitude column by removing trailing periods
    data["Latitude"] = data["Latitude"].str.rstrip('.')
    data["Latitude"] = [float(x) for x in data["Latitude"]]

    coords = data[["Latitude", "Longitude"]]
    kms_per_radian = 6371.0088
    epsilon = 85/ kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    cluster_labels = db.labels_
    data["Cluster"] = cluster_labels
    
    outliers = data.groupby("Cluster")[["Cluster","Latitude", "Longitude"]].count()
    i = outliers[(outliers.Cluster < 4)].index

        
    new_df = data.groupby("Cluster")[["Cluster","Latitude", "Longitude"]].median()
    new_df.drop(i, inplace=True)
    
    cluster_sizes = []

    for count in outliers.Cluster:
        if count >= 4:
            cluster_sizes.append(count)


    return new_df, cluster_sizes

new_df, cluster_sizes = location_cluster()

def color_coding(attribute):
    # COLOR BY TYPE
    if attribute == "Shark.common.name":
        shark_types = data["Shark.common.name"].unique()
        color_palette = sns.color_palette("Set1", len(shark_types))  # Use seaborn color palette\
        color_map = {shark_types[i]: mcolors.to_hex(color_palette[i]) for i in range(len(shark_types))}
        data["Shark.color"] = data["Shark.common.name"].map(color_map)

    if attribute == "Injury.severity":
        injury_types = data["Injury.severity"].unique()
        color_palette = sns.color_palette("Set1", len(injury_types))  # Use seaborn color palette\
        color_map = {injury_types[i]: mcolors.to_hex(color_palette[i]) for i in range(len(injury_types))}
        data["Injury.color"] = data["Injury.severity"].map(color_map)
    
    return color_map

def initial_fig_clustered():
    # Create the base map
    fig = go.Figure()

    # Add shark incident points
    fig.add_trace(go.Scattermapbox(
        lat=new_df["Latitude"],
        lon=new_df["Longitude"],
        mode="markers+text",
        marker=dict(size=15, color='blue'),
        text=cluster_sizes,  # Hover text
        textposition="middle center",  # Position of the text relative to the markers
        textfont=dict(color='white'),
        hovertext="Cick to see the details",  # Text displayed on hover
        hoverinfo="text",  # Text displayed on hover
        customdata=data.index,  # Pass row indices as custom data

    ))

    # Final map settings
    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",  # Disable the base map (shows no other lands)
            center={"lat": -23.69748, "lon": 133.88362},  # Center the map on Australia
            zoom=3.5,
        ),
        margin={"l": 0, "r": 0, "t": 0, "b": 0},  # Remove margins
        dragmode=False,
        showlegend=False,
    )
    return fig

fig = initial_fig_clustered()

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
                                        dcc.Graph(id='bar-chart', style={'marginBottom': '10px'}),
                                        
                                ]),
                                html.Div(id='previous-dropdown-value', style={'display': 'none'}),
                                dmc.Grid( justify="center", align="center", 
                                    children=[
                                        dmc.Col(
                                            span=6,
                                            children=[
                                                dmc.Switch(
                                                    id="switch_log_scale",
                                                    label="Log Scale",
                                                ),
                                            ],
                                        ),
                                        dmc.Col(
                                            span=6,
                                            children=[
                                                dmc.Switch(
                                                    id="switch_comparison",
                                                    label="COMPARE",
                                                ),
                                            ],
                                        ),
                                        ]

                                ),
                                html.Button(
                                    "Reset Zoom", 
                                    id="reset_button", 
                                    n_clicks=0,
                                    style={
                                        "marginTop": "20px",
                                        "backgroundColor": "red",  # Optional: Add background color for visibility
                                        "border": "none",
                                        "padding": "10px",
                                        "color": "white",
                                        "width": "25%",
                                        "align": "center",
                                    }
                                ),
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
                            style={"flex": "1", "overflow": "hidden","padding": 0, "margin": 0, "height":"100vh", "width":"100w"},  # Map resizes with the sidebar
                        ),
                    ],
                ),
            ],
        ),
    ],
)

# Global variable to store clicked categories and their colors
clicked_categories = []
bar_colors = []

# Callback to update the bar chart 
@app.callback(
    Output('bar-chart', 'figure'),
    [Input('dropdown-axis-bar', 'value'), Input('filter-options-bar', 'value'),Input("switch_log_scale", "checked")]
)
def update_bar_chart(selected_attribute, filter_option, log_scale):
    global bar_colors

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
        # colors = [f"rgb({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)})" for _ in range(len(filtered_counts))]
        category_order = 'total descending'
    elif filter_option == 'bottom_10':
        filtered_counts = counts.nsmallest(10, 'Occurrences')  # Bottom 10
        # colors = [f"rgb({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)})" for _ in range(len(filtered_counts))]
        category_order = 'total ascending'
    else:  # Show all
        filtered_counts = counts
        # colors = [f"rgb({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)})" for _ in range(len(filtered_counts))]
        category_order = 'total descending'

    bar_colors = filtered_counts[selected_attribute_renamed].map(color_coding(selected_attribute))

    # Create a bar chart using Plotly Graph Objects
    fig = go.Figure(data=[
        go.Bar(
            x=filtered_counts[selected_attribute_renamed],  # Set x-axis values
            y=filtered_counts['Occurrences'],  # Set y-axis values
            marker_color= bar_colors,  # Set the colors of the bars (if needed)
            customdata= bar_colors,  # Pass the colors as custom data
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



# Callback to handle all interactions with the map
@app.callback(
    [Output("shark-map", "figure"),
     Output("previous-dropdown-value", "children"),
     Output('bar-chart', 'clickData'),
     Output('shark-map', 'clickData')],
    [Input("bar-chart", "clickData"),
     Input("dropdown-axis-bar", "value"),
     Input("shark-map", "clickData"), 
     Input('switch_comparison', 'checked'),
     Input("reset_button", "n_clicks")],
    [State("previous-dropdown-value", "children")]
)
def handle_map_interactions(bar_click_data, selected_attribute, map_click_data, compare, reset_clicks, previous_attribute):
    global clicked_categories
    global fig
    ctx = dash.callback_context

    # If no triggered inputs, return the initial state
    if not ctx.triggered:
        return fig, selected_attribute, None, None

    # Determine which input triggered the callback
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

     # Define a dictionary to map original column names to desired names
    attribute_color_mapping = {
        'Injury.severity': 'Injury.color',
        'Shark.common.name': 'Shark.color',
        'Victim.activity': 'Victim.color',
        # Add more mappings as needed
    }

    #map the attribute to the color column
    color_column = attribute_color_mapping.get(selected_attribute, selected_attribute)

    # Handle map marker click for zooming into a location
    if triggered_id == "shark-map" and map_click_data:
        # Extract latitude and longitude of the clicked marker
        lat = map_click_data["points"][0]["lat"]
        lon = map_click_data["points"][0]["lon"]

        # fig = go.Figure()
        fig.data = []
        # Add trace for all points (if needed)
        fig.add_trace(go.Scattermapbox(
            lat=data["Latitude"],
            lon=data["Longitude"],
            mode="markers",
            marker=dict(size=15, color= data[color_column]),
            text=data["Injury.severity"],  # Hover text
            customdata=data.index,  # Pass row indices as custom data
        ))
        
        # Update map layout to zoom into the clicked point
        fig.update_layout(
            mapbox=dict(
                center={"lat": lat, "lon": lon},  # Center on clicked marker
                zoom=8  # Zoom level
            ),
            showlegend=False,
            dragmode=False
        )
        return fig, selected_attribute, None, None

    # Handle bar-chart click for filtering map points
    elif triggered_id == "bar-chart" and bar_click_data:
        # Check current zoom level to determine if bar chart interaction is allowed
        current_zoom = fig.layout.mapbox.zoom if "mapbox" in fig.layout else 3.5
        if current_zoom < 8:  # If zoom level is less than 8, ignore bar chart interactions
            return fig, selected_attribute, None, None

        if previous_attribute is None:
            previous_attribute = selected_attribute

        # Reset if the attribute changes or no data
        if bar_click_data is None or selected_attribute != previous_attribute:
            clicked_categories = []
            return fig, selected_attribute, None, None

        # Filter map points based on bar chart selection
        clicked_category = bar_click_data['points'][0]['x']
        marker_color = bar_click_data['points'][0]['customdata']  # Get color from bar

        # Add the clicked category and its color to the list if compare is checked
        if compare:
            if (clicked_category, marker_color) in clicked_categories:
                clicked_categories.remove((clicked_category, marker_color))
            else:
                clicked_categories.append((clicked_category, marker_color))
        else:
            if (clicked_category, marker_color) in clicked_categories:
                clicked_categories = []
            else:
                clicked_categories = [(clicked_category, marker_color)]

        # If clicked_categories is empty, return the initial figure
        if not clicked_categories:
            return fig, selected_attribute, None, None
    
        new_fig = go.Figure()

        # Plot all clicked categories
        for category, color in clicked_categories:
            filtered_map_df = data[data[selected_attribute] == category]
            new_fig.add_trace(go.Scattermapbox(
                lat=filtered_map_df['Latitude'],
                lon=filtered_map_df['Longitude'],
                mode='markers',
                marker=go.scattermapbox.Marker(size=15, color=color),
                text=filtered_map_df['Reference'],
                showlegend=False
            ))
        new_fig.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                zoom=current_zoom,  # Keep the current zoom level
                center={"lat": fig.layout.mapbox.center.lat, "lon": fig.layout.mapbox.center.lon},  # Keep current center
            ),
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            dragmode=False
        )
        return new_fig, selected_attribute, None, None

    # Handle reset button click
    elif triggered_id == "reset_button" and reset_clicks:
        
        # Remove all traces except the initial one
        fig= initial_fig_clustered()

        # Reset map to default view
        fig.update_layout(
            mapbox=dict(
                center={"lat": -23.69748, "lon": 133.88362},  # Default center
                zoom=3.5
            ),
            dragmode=False
        )
        return fig, selected_attribute , None, None

    # Default return
    return fig, selected_attribute, None, None

if __name__ == "__main__":
    app.run_server(debug=True)