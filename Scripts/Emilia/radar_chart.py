import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from math import log1p

# Your data
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
    "Provoked/Unprovoked Ratio": [0.24384787472035793,0.42105263157894735,0.38482384823848237,0.32098765432098764,0.5714285714285714,0.2535211267605634,0.3933649289099526],
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
#df["Normalized Ratio"] = log_transform_and_normalize(df["Provoked/Unprovoked Ratio"])
df["Normalized Tourism"] = log_transform_and_normalize(df["Tourism"])
df["Normalized Coastline"] = log_transform_and_normalize(df["Coastline"])

# Compute global maximums for consistent scaling
global_max = {
    "Normalized Incidents": df["Normalized Incidents"].max(),
    "Normalized Population": df["Normalized Population"].max(),
    #"Normalized Ratio": df["Normalized Ratio"].max(),
    "Normalized Tourism": df["Normalized Tourism"].max(),
    "Normalized Coastline": df["Normalized Coastline"].max()
}

# Initialize the Dash app
app = Dash(__name__)

# Function to create the overview radar chart
def create_radar_chart(categories, values_list, labels):
    fig = go.Figure()
    angles = categories + [categories[0]]

    for values, label in zip(values_list, labels):
        values += [values[0]]  # Complete the circle
        fig.add_trace(go.Scatterpolar(r=values, theta=angles, fill='toself', name=label))

    # Consistent scaling for all axes
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])  # Enforce consistent scaling
        ),
        showlegend=True
    )
    return fig

# Function to create a single state radar chart
def create_single_state_radar(state_data, state_name):
    categories = ["Normalized Incidents", "Normalized Population", "Normalized Tourism", "Normalized Coastline", "Provoked/Unprovoked Ratio"]
    values = state_data[categories].values.flatten().tolist() + [state_data[categories].values[0]]  # Complete the circle

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories + [categories[0]],
        fill='toself',
        name=state_name
    ))

    # Consistent scaling for all axes
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])  # Enforce consistent scaling
        ),
        showlegend=False,
        title=f"Attributes for {state_name}"
    )
    return fig

# App layout
app.layout = html.Div([
    html.H1("Radar Chart Viewer", style={"textAlign": "center"}),
    
    # First radar chart (overview)
    html.Div([
        html.H3("Overview Radar Chart", style={"textAlign": "center"}),
        dcc.Checklist(
            id="categories-checklist",
            options=[
                {"label": "Incidents", "value": "Normalized Incidents"},
                {"label": "Population", "value": "Normalized Population"},
                {"label": "Provoked/Unprovoked Ratio", "value": "Provoked/Unprovoked Ratio"},
                {"label": "Tourism", "value": "Normalized Tourism"},
                {"label": "Coastline", "value": "Normalized Coastline"}
            ],
            value=["Normalized Incidents", "Normalized Population"],
            inline=True
        ),
        dcc.Graph(id="radar-chart")
    ]),

    # Second radar chart (single state analysis)
    html.Div([
        html.H3("Single State Radar Chart", style={"textAlign": "center"}),
        dcc.Dropdown(
            id="state-dropdown",
            options=[{"label": state, "value": state} for state in df["State"]],
            value="NSW",
            multi=True,
            style={"width": "50%", "margin": "0 auto"}
        ),
        dcc.Graph(id="single-state-radar-chart")
    ])
])

# Callback for the overview radar chart
@app.callback(
    Output("radar-chart", "figure"),
    Input("categories-checklist", "value")
)
def update_radar_chart(selected_categories):
    categories = df["State"].tolist()
    values_list = [df[cat].tolist() for cat in selected_categories]
    labels = [cat.replace("Normalized ", "") for cat in selected_categories]
    return create_radar_chart(categories, values_list, labels)

# Callback for the single state radar chart
@app.callback(
    Output("single-state-radar-chart", "figure"),
    Input("state-dropdown", "value")
)
def update_single_state_radar(selected_state):
    state_data = df[df["State"] == selected_state].iloc[0]
    return create_single_state_radar(state_data, selected_state)

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
