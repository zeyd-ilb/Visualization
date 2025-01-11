import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output, State
import dash_mantine_components as dmc
from sklearn.cluster import DBSCAN
import numpy as np
from dash import Input, Output, State
import seaborn as sns
import matplotlib.colors as mcolors
from math import log1p


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

def attribute_preprocessing():
    # PREPROCESSING
    # Fill missing values with a placeholder for better readability and smooth mapping
    data["Injury.severity"] = data["Injury.severity"].fillna("Missing Information")
    data["Shark.common.name"] = data["Shark.common.name"].fillna("Missing Information")
    data["Location"] = data["Location"].fillna("Missing Information")
    data["Victim.activity"] = data["Victim.activity"].fillna("Missing Information")

attribute_preprocessing()

def color_coding(attribute):
    # COLOR BY TYPE
    if attribute == "Injury.severity":
        injury_types = data[attribute] .unique()
        color_palette = sns.color_palette("Set1", len(injury_types))  # Use seaborn color palette\
        color_map = {injury_types[i]: mcolors.to_hex(color_palette[i]) for i in range(len(injury_types))}
        data["Injury.color"] = data[attribute] .map(color_map)

    if attribute == "Shark.common.name":
        shark_types = data[attribute].unique()
        color_palette = sns.color_palette("Set1", len(shark_types))  # Use seaborn color palette\
        color_map = {shark_types[i]: mcolors.to_hex(color_palette[i]) for i in range(len(shark_types))}
        data["Shark.color"] = data[attribute].map(color_map)

    if attribute == "Victim.activity":
        shark_types = data[attribute].unique()
        color_palette = sns.color_palette("Set1", len(shark_types))  # Use seaborn color palette\
        color_map = {shark_types[i]: mcolors.to_hex(color_palette[i]) for i in range(len(shark_types))}
        data["Victim.color"] = data[attribute].map(color_map)

    return color_map

once = True
def annotation_shape():
    # Add an annotation
    fig.add_annotation(
        text="Please Select an Attribute First",
        xref="paper", yref="paper",
        x=0.25, y=0.92, showarrow=False,
        font=dict(size=24, color="white"),
        align="center",
    )

    # Add a shape
    fig.add_shape(
        type="rect",
        x0=0.20, y0=0.87, x1=0.80, y1=0.92,
        xref="paper", yref="paper",
        line=dict(color="RoyalBlue"),
        fillcolor="LightSkyBlue",
        opacity=0.5
    )

    return fig

def initial_fig_clustered():
    # Create the base map
    fig = go.Figure()

    # Add shark incident points
    fig.add_trace(go.Scattermapbox(
        lat=new_df["Latitude"],
        lon=new_df["Longitude"],
        mode="markers+text",
        marker=dict(size=20, color='blue'),
        text=cluster_sizes,  # Hover text
        textposition="middle center",  # Position of the text relative to the markers
        textfont=dict(color='white'),
        hovertext="Please select an attribute first",  # Text displayed on hover
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
fig = annotation_shape()    

def initial_line_chart():
    line_chart = go.Figure()

    # Add an annotation to display the initial message
    line_chart.add_annotation(
        text="Click on the bar chart to see the yearly progress",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20, color="white"),
        align="center"
    )

    # Optionally, you can set the layout to ensure the annotation is centered
    line_chart.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="black",
        paper_bgcolor="black",
    )
    return line_chart

line_chart = initial_line_chart()

def attribute_rename(selected_attribute):
    # Define a dictionary to map original column names to desired names
    attribute_name_mapping = {
        'Injury.severity': 'Injury Severity',
        'Shark.common.name': 'Shark Species',
        'Victim.activity': 'Victim Activity',
        # Add more mappings as needed
    }
    # Update the selected_attribute if it is one of the renamed columns
    selected_attribute_renamed = attribute_name_mapping.get(selected_attribute, selected_attribute)

    return selected_attribute_renamed, attribute_name_mapping

def parallel_chart():
    
    # Define the data directly within the file
    data = {
        "State": ["NSW", "NT", "QLD", "SA", "TAS", "VIC", "WA"],
        "Number of Incidents": [449, 19, 372, 81, 28, 71, 213],
        "Population": [8469.6, 254.3, 5560.5, 1873.8, 575.7, 6959.2, 2951.6],
        "Tourism": [3702, 202, 2124, 451, 256, 2489, 819],
        "Coastline": [2137, 10953, 13347, 5067, 4882, 2512, 20781]
    }

    ratio_data = {
        "State": ["NSW", "NT", "QLD", "SA", "TAS", "VIC", "WA"],
        "Provoked": [109, 8, 142, 26, 16, 18, 83],
        "Unprovoked": [338, 11, 227, 55, 12, 53, 128],
        "Provoked/Unprovoked Ratio": [0.24384787472035793, 0.42105263157894735, 0.38482384823848237, 0.32098765432098764, 0.5714285714285714, 0.2535211267605634, 0.3933649289099526],
    }

    df = pd.DataFrame(data)
    ratio_df = pd.DataFrame(ratio_data)

    # Merge the datasets
    df = pd.merge(df, ratio_df, on="State")

    # Log transformation and normalization
    def log_transform_and_normalize(series):
        log_transformed = series.apply(log1p)  # log1p ensures log(0+1) = 0
        return log_transformed / log_transformed.max()

    df["Normalized Incidents"] = log_transform_and_normalize(df["Number of Incidents"])
    df["Normalized Population"] = log_transform_and_normalize(df["Population"])
    df["Normalized Tourism"] = log_transform_and_normalize(df["Tourism"])
    df["Normalized Coastline"] = log_transform_and_normalize(df["Coastline"])

    # Define the columns to plot
    normalized_columns = [
        "Normalized Incidents", 
        "Normalized Population", 
        "Normalized Tourism", 
        "Normalized Coastline"
    ]

    # Create the line plot
    line_fig = go.Figure()

    for state in df["State"].unique():
        state_data = df[df["State"] == state]
        line_fig.add_trace(go.Scatter(
            x=normalized_columns,
            y=state_data[normalized_columns].values[0],
            mode='lines+markers',
            name=state
        ))

    # Update layout
    line_fig.update_layout(
        template="plotly_dark",
        title="Parallel Coordinate Plot of States",
        xaxis_title="Metrics",
        yaxis_title="Normalized Values",
        xaxis=dict(tickmode='array', tickvals=list(range(len(normalized_columns))), ticktext=normalized_columns),
        legend_title="State"
    )

    return line_fig  

line_fig = parallel_chart()

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
                                "padding": "15px",
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
                                html.H1("Shark Incident Details", style={"marginBottom": "10px", "color": "#fff"}),  # Light text color
                                html.Label("Select Attribute for Bar Chart:", style={'color': '#fff'}),  # Light text color
                                dcc.Dropdown(
                                    id='dropdown-axis-bar',
                                    options=[
                                        # {
                                        #     "label": html.Span(['Select an Attribute'], style={'color': '#fff', 'backgroundColor': '#000'}),  
                                        #     "value": 'nothing',
                                        # },
                                        {
                                            "label": html.Span(['Injury Severity'], style={'color': '#fff', 'backgroundColor': '#000'}),  
                                            "value": 'Injury.severity',
                                        },
                                        {
                                            "label": html.Span(['Shark Species'], style={'color': '#fff', 'backgroundColor': '#000'}),
                                            "value": 'Shark.common.name',
                                        },
                                        {
                                            "label": html.Span(['Victim Activity'], style={'color': '#fff', 'backgroundColor': '#000'}),
                                            "value": 'Victim.activity',
                                        }
                                    ],
                                    value=None,  # Default value
                                    clearable=False,
                                    style={
                                        'marginTop': '5px',
                                        'marginBottom': '20px',
                                        'color': '#fff',  # Light text color
                                        'backgroundColor': '#000',  # Black background color
                                        'width': '75%',
                                        'height'   : '40px',
                                    }
                                ),
                                #BAR CHART
                                html.Div(
                                    id='bar-chart-container',
                                    style={'display':'none'},
                                    children=[

                                        html.Label("Filter Options:", style={'marginBottom': '10px', "color": "#fff"}),  # Light text color
                                        dmc.SegmentedControl(
                                            id='filter-options-bar',
                                            style = {'marginBottom': '25px', "color": "#fff"},
                                            orientation="horizontal",
                                            fullWidth=True,
                                            value='top_10',  # Default option
                                            data=[
                                                {'label': '10 Most Frequent', 'value': 'top_10'},
                                                {'label': '10 Least Frequent', 'value': 'bottom_10'},
                                                {'label': 'Show All', 'value': 'all'}
                                                ]
                                        ),
                                        dmc.Grid( justify="center", align="center", 
                                            style={"marginBottom": 15,"marginLeft": 5, "padding": 0},
                                            children=[
                                                dmc.Col(
                                                    span=4,
                                                    children=[
                                                        dmc.Switch(
                                                            id="switch_log_scale",
                                                            label="Log Scale",
                                                        ),
                                                    ],
                                                ),
                                                dmc.Col(
                                                    span=4,
                                                    children=[
                                                        dmc.Switch(
                                                            id="switch_comparison",
                                                            label="Select Multiple",
                                                        ),
                                                    ],
                                                ),
                                                dmc.Col(
                                                    span=4,
                                                    children=[
                                                        dmc.Button(
                                                            "Reset Zoom", 
                                                            id="reset_button", 
                                                            n_clicks=0,
                                                            variant="gradient",
                                                            gradient={"from": "teal", "to": "blue", "deg": 60},

                                                            style={
                                                                "color": "white",
                                                                "width": "100%",
                                                                "align": "center",
                                                            }
                                                        ),
                                                    ]
                                                )
                                            ]

                                        ),
                                        html.Label("Click on a bar to see the points:", style={'marginBottom': '10px', "color": "#fff"}), 
                                        dcc.Graph(id='bar-chart', style={'marginBottom': '10px'},config={'displayModeBar': False}),
                                        dcc.Graph(id='line-chart', style={'marginBottom': '10px'},config={'displayModeBar': False}),
                                        html.H1("Parallel Coordinate Plot for States"),
                                        dcc.Graph(id="parallel-coordinate-plot", figure=line_fig),
                                        
                                ]),
                                html.Div(id='previous-dropdown-value', style={'display': 'none'}),
                               
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
                            config={'displayModeBar': False},
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

@app.callback(
    Output('bar-chart-container', 'style'),
    Input('dropdown-axis-bar', 'value')
)
def update_visibility(selected_attribute):
    if selected_attribute:
        return {'display': 'block'}  # Show the division
    return {'display': 'none'}  # Hide the division


# Callback to update the bar chart 
@app.callback(
    Output('bar-chart', 'figure'),
    [Input('dropdown-axis-bar', 'value'), Input('filter-options-bar', 'value'),Input("switch_log_scale", "checked")]
)
def update_bar_chart(selected_attribute, filter_option, log_scale):
    global bar_colors

    if not selected_attribute:
        return go.Figure()
    
    # Count occurrences of each unique value in the selected attribute
    counts = data[selected_attribute].value_counts(dropna=False).reset_index()
    counts.columns = [selected_attribute, 'Occurrences']

    # Update the selected_attribute if it is one of the renamed columns
    selected_attribute_renamed, column_name_mapping = attribute_rename(selected_attribute)
    
    # Rename the columns in the DataFrame
    counts.rename(columns=column_name_mapping, inplace=True)

    # Apply filtering based on the selected filter option
    if filter_option == 'top_10':
        filtered_counts = counts.nlargest(10, 'Occurrences')  # Top 10
        category_order = 'total descending'
    elif filter_option == 'bottom_10':
        filtered_counts = counts.nsmallest(10, 'Occurrences')  # Bottom 10
        category_order = 'total ascending'
    else:  # Show all
        filtered_counts = counts
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
     Output('shark-map', 'clickData'),
     Output('line-chart', 'figure')],
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
    global line_chart
    global once

    ctx = dash.callback_context

    # If no attribute is selected, return the initial state
    if selected_attribute is None: 
        return fig, None, None, None, line_chart  
    if once:
        fig = initial_fig_clustered()
        once = False    
    
        # Update the hover text
        fig.update_traces(go.Scattermapbox(
            hovertext="Click on to zoom",  # Text displayed on hover
            marker=dict(size=20, color='#5506d6'),
        ))

    # If no triggered inputs, return the initial state
    if not ctx.triggered:
        return fig, selected_attribute, None, None, line_chart

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

    if previous_attribute is None:
        previous_attribute = selected_attribute

    # Reset if the attribute changes or no data
    if selected_attribute != previous_attribute:
        clicked_categories = []
        # fig.data = []

        # Recolor the shark points based on the new attribute
        fig.add_trace(go.Scattermapbox(
            lat=data["Latitude"],
            lon=data["Longitude"],
            marker=dict(size=15, color= data[color_column]),
            text=data[selected_attribute],  # Hover text
            customdata=data.index,  # Pass row indices as custom data
        ))

        return fig, selected_attribute, None, None, line_chart

    # Handle map marker click for zooming into a location
    if triggered_id == "shark-map" and map_click_data:

        # Extract latitude and longitude of the clicked marker
        lat = map_click_data["points"][0]["lat"]
        lon = map_click_data["points"][0]["lon"]

        fig.data = []
        # Add trace for all points (if needed)
        fig.add_trace(go.Scattermapbox(
            lat=data["Latitude"],
            lon=data["Longitude"],
            mode="markers",
            marker=dict(size=15, color= data[color_column]),
            text=data[selected_attribute],  # Hover text
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
        return fig, selected_attribute, None, None, line_chart

    # Handle bar-chart click for filtering map points
    elif triggered_id == "bar-chart" and bar_click_data:
       
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
            return fig, selected_attribute, None, None, line_chart
    
        selected_attribute_renamed = attribute_rename(selected_attribute)[0]

        line_chart = go.Figure()
        
        # Plot all clicked categories
        for category, color in clicked_categories:
            filtered_map_df = data[data[selected_attribute] == category]
            yearly_counts = filtered_map_df.groupby('Incident.year').size().reset_index(name='Occurrences')

            line_chart.add_trace(go.Scatter(
                x=yearly_counts['Incident.year'],
                y=yearly_counts['Occurrences'],
                mode='lines+markers',
                name=category,
                hovertemplate='%{x}<br>Occurrences: %{y}<extra></extra>',
                line=dict(color=color),
                marker=dict(color=color)
            ))
        
        # Update the line chart layout
        line_chart.update_layout(
            title=f"Occurrences of {selected_attribute_renamed} Over Years",
            xaxis=dict(
                title="Year",
                type="linear",  # Keep the x-axis linear for years
            ),
            yaxis=dict(
                title="Number of Occurrences",
                type="linear",  # Apply log scale if enabled
            ),
            template='plotly_dark',  # Set the dark theme
            legend=dict(
                x=0.2,  # Horizontal position (0 to 1)
                y=1,    # Vertical position (0 to 1)
                xanchor='center',  # Anchor the legend horizontally at the center
                yanchor='top',     # Anchor the legend vertically at the top
                bgcolor='rgba(0,0,0,0)',  # Background color of the legend (transparent)
                bordercolor='rgba(255,255,255,0.5)',  # Border color of the legend
                borderwidth=1  # Border width of the legend
            )
        )

        # Check current zoom level to determine if bar chart interaction is allowed
        current_zoom = fig.layout.mapbox.zoom if "mapbox" in fig.layout else 3.5
        if current_zoom < 8:  # If zoom level is less than 8, ignore bar chart interactions
            return fig, selected_attribute, None, None, line_chart

        new_fig = go.Figure()
        
        # Plot all clicked categories
        for category, color in clicked_categories:
            filtered_map_df = data[data[selected_attribute] == category]
            yearly_counts = filtered_map_df.groupby('Incident.year').size().reset_index(name='Occurrences')

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
        

        return new_fig, selected_attribute, None, None, line_chart

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
        return fig, selected_attribute , None, None, line_chart

    # Default return
    return fig, selected_attribute, None, None, line_chart



if __name__ == "__main__":
    app.run_server(debug=True)