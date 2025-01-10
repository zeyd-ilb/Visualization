import dash
from dash import dcc, html
import pandas as pd
from math import log1p
import plotly.express as px

# Data initialization
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

# Map states to numerical values
state_to_numeric = {state: i for i, state in enumerate(df["State"].unique())}
df["State Numeric"] = df["State"].map(state_to_numeric)

# Prepare data for parallel coordinates
normalized_columns = [
    "Normalized Incidents", 
    "Normalized Population", 
    "Normalized Tourism", 
    "Normalized Coastline"
]

# Add a combined column for Dash
plot_df = df[["State", "State Numeric"] + normalized_columns]

# Dash app initialization
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Parallel Coordinate Plot for States"),
    dcc.Graph(
        id="parallel-coordinate-plot",
        figure=px.parallel_coordinates(
            plot_df,
            dimensions=normalized_columns,
            color="State Numeric",  # Use the numerical mapping for color
            color_continuous_scale=px.colors.sequential.Viridis,  # Color scale
            labels={
                "Normalized Incidents": "Incidents",
                "Normalized Population": "Population",
                "Normalized Tourism": "Tourism",
                "Normalized Coastline": "Coastline",
                "State Numeric": "State"
            },
            title="Parallel Coordinate of State Outcomes"
        )
    )
])

if __name__ == "__main__":
    app.run_server(debug=True)
