import pandas as pd
import matplotlib.pyplot as plt
from math import log1p

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

# Create the parallel coordinate plot
plt.figure(figsize=(12, 6))

# Select normalized columns for plotting
columns_to_plot = [
    "Normalized Incidents", 
    "Normalized Population", 
    "Normalized Tourism", 
    "Normalized Coastline"
]

# Plot each state as a line
for _, row in df.iterrows():
    plt.plot(columns_to_plot, row[columns_to_plot], label=row["State"], marker='o')

# Labeling
plt.xticks(range(len(columns_to_plot)), columns_to_plot, rotation=45)
plt.title("Parallel Coordinate Plot of States")
plt.xlabel("Metrics")
plt.ylabel("Normalized Values")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Show the plot
plt.show()
