import dash
import dash_mantine_components as dmc

from components.charts import chart_example
from components.functionality_selection import functionality_selection
from components.input_environmental_personal import input_environmental_personal
from components.dropdowns import model_selection, chart_selection
from components.my_card import my_card
from utils.my_config_file import (
    URLS,
    ElementsIDs,
)
from dash import html, dcc

dash.register_page(__name__, path=URLS.HOME.value)

layout = dmc.Stack(
    [
        dmc.Grid(
            children=[
                dmc.GridCol(
                    model_selection(),
                    span={"base": 12, "sm": 6},
                ),
                dmc.GridCol(
                    functionality_selection(),
                    span={"base": 12, "sm": 6},
                ),
            ],
            gutter="xl",
        ),
        dmc.Grid(
            children=[
                my_card(
                    title="",
                    children=input_environmental_personal(''),
                    id = 'input_card',
                    span={"base": 12, "sm": 3},
                ),
                my_card(
                    title="Results and Visualization",
                    children=dmc.Stack(
                        [
                            chart_selection(),
                            dmc.SimpleGrid(
                                cols=3,
                                spacing="md",
                                verticalSpacing="md",
                                id = 'graph-container',
                                children=[
                                    dmc.Center(dmc.Text("PMV = 0.5")),
                                    dmc.Center(
                                        dmc.Text("PPD = 2"),
                                    ),
                                    dmc.Center(
                                        dmc.Text("SET = 30.1"),
                                    ),
                                    dmc.Center(
                                        dmc.Text("Result 4"),
                                    ),
                                    dmc.Center(
                                        dmc.Text("Result 5"),
                                    ),
                                ],
                            ),
                            dcc.Graph(
                                id=ElementsIDs.CHART_CONTAINER.value,
                                figure=chart_example(),
                            ),
                        ],
                    ),
                    
                    span={"base": 12, "sm": 9},
                ),
            ],
            gutter="xl",
        ),
    ]
)
