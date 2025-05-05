import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd
from database import connect_db
import requests

collection = connect_db()
projection = {
    "Assessments.Primary Problem": 1,
    "Place.State Reference": 1,
    "Time / Day.Date": 1
}

df = pd.json_normalize(collection.find({}, projection))


def init_dash(server):
    dash_app = dash.Dash(
        __name__,
        server=server,
        url_base_pathname='/dash_app/',
        suppress_callback_exceptions=True
    )

    # Fig Time
    df["Time / Day.Date"] = df["Time / Day.Date"].dt.normalize()  # Remove time component
    df_grouped = df.groupby("Time / Day.Date").size().reset_index(name="count")
    fig_time = px.line(df_grouped, x='Time / Day.Date', y='count', title="Reports Over Time")

    # Fig Problem
    df_grouped = df.groupby("Assessments.Primary Problem").size().reset_index(name="count")
    fig_problem = px.bar(df_grouped, x='Assessments.Primary Problem', y='count', title="Reports By Primary Problem")

    # Fig States
    states_count = df.groupby("Place.State Reference").size().reset_index(name="count")
    geojson_url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    response = requests.get(geojson_url)
    geojson_states = response.json()

    fig_states = px.choropleth(states_count,
                               locations='Place.State Reference',  # Column with state abbreviations
                               locationmode='USA-states',  # This works with both names and abbreviations
                               color='count',
                               scope='usa',
                               color_continuous_scale='Burg',
                               title='Reports by US States')

    dash_app.layout = html.Div([
        html.H2("Total Reports: " + str(df.shape[0])),
        dcc.Graph(figure=fig_time),
        dcc.Graph(figure=fig_problem),
        dcc.Graph(figure=fig_states)
    ])

    """
    df_test = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas"],
        "Amount": [10, 15, 7]
    })
    fig = px.bar(df_test, x="Fruit", y="Amount", title="Fruit Count")

    dash_app.layout = html.Div([
        html.H2("This is the Dash App"),
        dcc.Graph(figure=fig)
    ])
    """
