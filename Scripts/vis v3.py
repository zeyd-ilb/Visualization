#interactive shark map - bar chart
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
#TO NAME THE STATES
"""fig.add_trace(go.Choroplethmapbox(
    geojson=geojson_data,
    featureidkey="properties.STATE_NAME",
    locations=[],  # No data binding to states
    z=[],  # No color data
    showscale=False,  # Disable the color scale
))"""

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
app.layout = html.Div(
    style={"display": "flex", "height": "100vh", "overflow": "hidden", "width": "100%" },
    children=[
        html.Div(
            id="resizable-sidebar",
            style={
                "width": "700px",  # Initial sidebar width
                "minWidth": "250px",  # Minimum width
                "maxWidth": "750px",  # Maximum width
                "padding": "30px",
                "borderRight": "1px solid #444",  # Darker border color
                "backgroundColor": "#000",  # Black background color
                "color": "#fff",  # Light text color
                "display": "flex",
                "flexDirection": "column",
                "overflowY": "auto",
                "resize": "horizontal",
                "overflow": "auto",
            },
            children=[
                html.H2("Shark Incident Details", style={"marginBottom": "10px", "color": "#fff"}),  # Light text color
                html.Div("Click on a point to see details.", id="incident-details", style={"marginBottom": "20px", "color": "#fff"}),  # Light text color
                
                # Bar chart section
                html.Div([
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
                            'color': '#fff',  # Light text color
                            'backgroundColor': '#000',  # Black background color
                        }
                    ),
                    html.Label("Filter Options:", style={'marginBottom': '10px', "color": "#fff"}),  # Light text color
                    dcc.RadioItems(
                        id='filter-options-bar',
                        options=[
                            {'label': 'Top 10 Most Frequent', 'value': 'top_10'},
                            {'label': 'Bottom 10 Least Frequent', 'value': 'bottom_10'},
                            {'label': 'Show All', 'value': 'all'}
                        ],
                        value='top_10',  # Default option
                        style={'marginBottom': '10px', "color": "#fff"}  # Light text color
                    ),
                    dcc.Graph(id='bar-chart', style={'margin': 'auto'})
                ]),
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
}"""


# Callback to update the bar chart 
@app.callback(
    Output('bar-chart', 'figure'),
    [Input('dropdown-axis-bar', 'value'), Input('filter-options-bar', 'value')]
)
def update_bar_chart(selected_attribute, filter_option):
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
    selected_attribute = column_name_mapping.get(selected_attribute, selected_attribute)

    # Apply filtering based on the selected filter option
    if filter_option == 'top_10':
        filtered_counts = counts.nlargest(10, 'Occurrences')  # Top 10
    elif filter_option == 'bottom_10':
        filtered_counts = counts.nsmallest(10, 'Occurrences')  # Bottom 10
    else:  # Show all
        filtered_counts = counts

    # Create a bar chart using Plotly Express
    fig = px.bar(
        filtered_counts,
        x=selected_attribute,
        y='Occurrences',
        title=f"{selected_attribute} by Frequency ({filter_option.replace('_', ' ').capitalize()})",
        labels={selected_attribute: selected_attribute, 'Occurrences': 'Number of Occurrences'},
        template='plotly_dark',
        color=selected_attribute  # Color bars based on the selected attribute
    )

    # Adjust layout for better readability
    fig.update_layout(xaxis={'categoryorder': 'total descending'})
    return fig



# Callback to update shark map based on bar click
@app.callback(
    Output('shark-map', 'figure'),
    Input('bar-chart', 'clickData')
)
def update_shark_map(clickData):
    if clickData is None:
        return fig

    clicked_category = clickData['points'][0]['x']
    filtered_map_df = data[data['Injury.severity'] == clicked_category]

    new_fig = go.Figure(go.Scattermapbox(
        lat=filtered_map_df['Latitude'],
        lon=filtered_map_df['Longitude'],
        mode='markers',
        marker=go.scattermapbox.Marker(size=8),
        text=data['Reference'],
    ))
    new_fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            zoom=3.8,  # Adjust zoom level to tightly frame Australia
            center={"lat": -25.0, "lon": 134.0},  # Center the map on Australia
        ),
        margin={"l": 0, "r": 0, "t": 0, "b": 0}
    )
    return new_fig


if __name__ == "__main__":
    app.run_server(debug=True)
