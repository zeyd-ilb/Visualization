# Import packages
from dash import Dash, html, dash_table, dcc
import pandas as pd
import plotly.express as px
import plotly.express as px

# Incorporate data
df = pd.read_csv('Australian Shark-Incident Database Public Version.csv')

# Initialize the app
app = Dash()

# App layout
app.layout = [
    html.Div(children='My First App with Data'),
    dcc.Graph(figure=px.histogram(df, x='Victim.activity', histfunc='count', color='Victim.gender'))

]

# Run the app
if __name__ == '__main__':
    app.run(debug=True)



def random_color():
return f"#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}"

colors = []
unique_shark_types = data['Shark.common.name'].unique(),
for i in unique_shark_types:
    colors.append(random_color())