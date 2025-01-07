import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Load the updated dataset
#data_path = r"C:\Users\emili\Downloads\JBI100 Data (2024-2025)\sharks.csv"
df = pd.read_csv('Australian Shark-Incident Database Public Version.csv')

# Standardize 'Injury.severity' column (removes leading/trailing spaces and converts to lowercase)
df['Injury.severity'] = df['Injury.severity'].str.strip().str.lower()
df['Injury.severity'] = df['Injury.severity'].replace('fatality', 'fatal')

# Initialize Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Shark Attack Data Analysis", style={'textAlign': 'center', 'marginBottom': '20px'}),
    
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
            style={'marginBottom': '20px'}
        ),
        dcc.Graph(id='bar-chart', style={'margin': 'auto'})
    ], style={'width': '50%', 'margin': 'auto', 'marginBottom': '40px'}),
    
    # Scatterplot section
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
    ], style={'width': '50%', 'margin': 'auto', 'marginBottom': '40px'}),
])


# Callback to update the bar chart based on dropdown selection
@app.callback(
    Output('bar-chart', 'figure'),
    [Input('dropdown-axis-bar', 'value')]
)
def update_bar_chart(selected_attribute):
    # Count occurrences of each unique value in the selected attribute
    counts = df[selected_attribute].value_counts().reset_index()
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

# Callback to update the scatter plot based on dropdown selection
@app.callback(
    Output('scatterplot', 'figure'),
    [Input('dropdown-axis-scatter', 'value')]
)
def update_scatter_plot(selected_x_axis):
    # Create a scatter plot with selected X-axis
    fig = px.scatter(
        df, 
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

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
