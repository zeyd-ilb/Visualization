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


# Load the shark attack data
data = pd.read_csv("C:\\Users\\20223070\\Downloads\\sharks.csv")

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
fig = go.Figure(initial_fig)

tabs_styles = {
    'height': '45px',
    'width': '100%',
    'margin': '0px',
    'padding': '0px',
    'border': '0px',
    'color': 'black',
    'backgroundColor': 'black',
    'fontWeight': 'bold',
    'fontSize': '20px',
    'fontFamily': 'Arial',
    'textAlign': 'center',
    'display': 'flex',
    'justifyContent': 'space-between',
    'alignItems': 'center',
    'borderLeft': '0px',
    'borderColor': 'black',
    'borderStyle': 'solid',
    'borderWidth': '0px',
    'borderLeftWidth': '0px',
    'borderTopColor': 'black',
    'borderLeftColor': 'black',
    'borderRightColor': 'black',
    'borderBottomColor': 'black',
    'borderTopStyle': 'solid',
    'borderLeftStyle': 'solid',
    'borderRightStyle': 'solid',
    'borderBottomStyle': 'solid',

}
tab_style = {
    # 'padding': '3px',
    # 'fontWeight': 'bold'
    'padding': '0px',
    # 'margin': '0px',
    'backgroundColor': 'black',
    'fontSize': '20px',
    'fontFamily': 'Arial',
    'textAlign': 'center',
    "color" : "white",
}

tab_selected_style = {
    'backgroundColor': 'black',
    'color': 'white',
    # 'padding': '0px',
    # 'margin': '0px',
    # 'fontWeight': 'bold',
}

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
    [Input('dropdown-axis-bar', 'value'), Input('filter-options-bar', 'value'),Input("switch_log_scale", "checked")]
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
# Global variable to store clicked categories and their colors
clicked_categories = []



# Callback to update shark map based on bar click
@app.callback(
    [Output('shark-map', 'figure'),
     Output('previous-dropdown-value', 'children'),
     Output('bar-chart', 'clickData')],
    [Input('bar-chart', 'clickData'),
     Input('switch_comparison', 'checked'),
     Input('dropdown-axis-bar', 'value')],
    [State('previous-dropdown-value', 'children')]
)
def update_shark_map(clickData, compare, selected_attribute, previous_attribute):
    global clicked_categories

    if previous_attribute is None:
        previous_attribute = selected_attribute
    
    # If no bar is clicked or the selected attribute is changed, return the initial figure
    if clickData is None or selected_attribute != previous_attribute:
        clicked_categories = []  
        return initial_fig, selected_attribute, None
    
    clicked_category = clickData['points'][0]['x']
    marker_color = clickData['points'][0]['customdata']  # Get the color of the clicked bar
    
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
        return initial_fig, selected_attribute, None
    
    new_fig = go.Figure()
    
    # Plot all clicked categories
    for category, color in clicked_categories:
        filtered_map_df = data[data[selected_attribute] == category]
        new_fig.add_trace(go.Scattermapbox(
            lat=filtered_map_df['Latitude'],
            lon=filtered_map_df['Longitude'],
            mode='markers',
            marker=go.scattermapbox.Marker(size=8, color=color),
            text=filtered_map_df['Reference'],
        ))

    new_fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            zoom=3.5,
            center={"lat": -25.0, "lon": 134.0},  # Center the map on Australia
        ),
        margin={"l": 0, "r": 0, "t": 0, "b": 0}
    )

    return new_fig, selected_attribute , None

if __name__ == "__main__":
    app.run_server(debug=True)