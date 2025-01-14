#version with slider
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import requests
import plotly.express as px


# Load the shark attack data
data = pd.read_csv("C:\\Users\\20223070\\Downloads\\sharks.csv")

# Load a GeoJSON file with only Australia boundaries
geojson_url = "https://raw.githubusercontent.com/rowanhogan/australian-states/master/states.geojson"
geojson_data = requests.get(geojson_url).json()

# Create the base map
fig = go.Figure()

# Add a GeoJSON layer to show only Australia's boundaries
fig.add_trace(go.Choroplethmapbox(
    geojson=geojson_data,
    featureidkey="properties.STATE_NAME",
    locations=[],  # No data binding to states
    z=[],  # No color data
    showscale=False,  # Disable the color scale
))

# Add shark incident points
fig.add_trace(go.Scattermapbox(
    lat=data["Latitude"],
    lon=data["Longitude"],
    mode="markers",
    marker=dict(size=8, color="red"),
    text=data["Reference"],  # Hover text
    hoverinfo="text",  # Text displayed on hover
    customdata=data.index,  # Pass row indices as custom data
))

# Final map settings
fig.update_layout(
    mapbox=dict(
        # style="white-bg",  # Disable the base map (shows no other lands)
        style="carto-darkmatter",  # Disable the base map (shows no other lands)
        center={"lat": -25.0, "lon": 134.0},  # Center the map on Australia
        zoom=3.8,  # Adjust zoom level to tightly frame Australia
        layers=[
            dict(
                sourcetype="geojson",
                source=geojson_data,  # Use GeoJSON for Australia's boundaries
                below="traces",
                type="fill",
                color="lightblue",  # Fill Australia's boundaries with light blue
            )
        ],
    ),
    margin={"l": 0, "r": 0, "t": 0, "b": 0},  # Remove margins
)

# Dash app
app = dash.Dash(__name__)

# App layout with a resizable sidebar
app.layout = html.Div(
    style={"display": "flex", "height": "98vh", "overflow": "hidden"},
    children=[
        html.Div(
            id="resizable-sidebar",
            style={
                "width": "700px",  # Initial sidebar width
                "minWidth": "250px",  # Minimum width
                "maxWidth": "750px",  # Maximum width
                "padding": "5px",
                "borderRight": "1px solid #ccc",
                "display": "flex",
                "flexDirection": "column",
                "overflowY": "auto",
                "resize": "horizontal",
                "overflow": "auto",
            },
            children=[
                html.H2("Shark Incident Details", style={"marginBottom": "10px"}),
                html.Div("Click on a point to see details.", id="incident-details", style={"marginBottom": "20px"}),
                dcc.Graph(id="sidebar-bar-chart", style={"flex": "1", "margin": "10px",'marginBottom':"10px"}),
                
                # Bar chart section
                    html.Div([
                        html.Label("Select Attribute for Bar Chart:", style={'marginBottom': '10px'}),
                        dcc.Dropdown(
                            id='dropdown-axis-bar',
                            options=[
                                {'label': 'Injury Severity', 'value': 'Injury.severity'},
                                {'label': 'Shark Species', 'value': 'Shark.common.name'},
                                {'label': 'Location', 'value': 'Location'},
                                {'label': 'Victim Activity', 'value': 'Victim.activity'}
                            ],
                            value='Injury.severity',  # Default value
                            clearable=False,
                            style={'marginBottom': '5px'}
                        ),
                        dcc.Graph(id='bar-chart', style={'margin': 'auto'})
                    ], style={'width': '100%', 'margin': 'auto', 'marginBottom': '5px'}),

            ],
        ),

        dcc.Graph(
            id="shark-map",
            figure=fig,
            style={"flex": "1", "overflow": "hidden"},  # Map resizes with the sidebar
        ),


    ],
)

# Callback to update side panel on point click
@app.callback(
    [Output("incident-details", "children"), Output("sidebar-bar-chart", "figure")],
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

    # Example bar chart: Number of incidents by "Fatal" status for demo purposes
    fatal_data = data["Shark.common.name"].value_counts()  # Replace with relevant data for bar chart
    bar_chart = go.Figure(
        data=[
            go.Bar(
                x=fatal_data.index,
                y=fatal_data.values,
                # marker_color=["green", "red"],  # Custom colors
            )
        ],
        layout=go.Layout(
            title=dict(text="Fatal vs Non-Fatal Incidents", x=0.5),  # Center title
            xaxis=dict(title="Status"),
            yaxis=dict(title="Count"),
            margin=dict(l=0, r=0, t=40, b=30),  # Small margins around the chart
            height=None,  # Remove fixed height to make it responsive
        )
    )

    return details, bar_chart

# Callback to update the bar chart based on dropdown selection
@app.callback(
    Output('bar-chart', 'figure'),
    [Input('dropdown-axis-bar', 'value')]
)
def update_bar_chart(selected_attribute):
    # Count occurrences of each unique value in the selected attribute
    counts = data[selected_attribute].value_counts().reset_index()
    counts.columns = [selected_attribute, 'Occurrences']

    # Create a bar chart using Plotly Express
    fig = px.bar(
        counts,
        x=selected_attribute,
        y='Occurrences',
        title=f"{selected_attribute} by Frequency",
        labels={selected_attribute: selected_attribute, 'Occurrences': 'Number of Occurrences'},
        template='plotly',
        color=selected_attribute  # Color bars based on the selected attribute
    )

    # Adjust layout for better readability
    fig.update_layout(xaxis={'categoryorder': 'total descending'})
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
