import base64
import io
from copy import deepcopy

import dash_mantine_components as dmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pythermalcomfort.models import pmv, adaptive_ashrae
from pythermalcomfort.utilities import v_relative, clo_dynamic
from scipy import optimize

from components.drop_down_inline import generate_dropdown_inline
from utils.my_config_file import (
    ElementsIDs,
    Models,
    Functionalities,
    UnitSystem,
    UnitConverter,
)
from utils.website_text import TextHome
import matplotlib
from pythermalcomfort.models import adaptive_en
from pythermalcomfort.psychrometrics import t_o

matplotlib.use("Agg")

import plotly.graph_objects as go
from dash import dcc


def chart_selector(selected_model: str, function_selection: str):
    list_charts = list(Models[selected_model].value.charts)
    if function_selection == Functionalities.Compare.value:
        if selected_model == Models.PMV_ashrae.name:
            list_charts = list(Models[selected_model].value.charts_compare)

    list_charts = [chart.name for chart in list_charts]
    drop_down_chart_dict = {
        "id": ElementsIDs.chart_selected.value,
        "question": TextHome.chart_selection.value,
        "options": list_charts,
        "multi": False,
        "default": list_charts[0],
    }

    return generate_dropdown_inline(
        drop_down_chart_dict, value=drop_down_chart_dict["default"], clearable=False
    )


def get_inputs(inputs):
    tr = inputs[ElementsIDs.t_r_input.value]
    t_db = inputs[ElementsIDs.t_db_input.value]
    met = inputs[ElementsIDs.met_input.value]
    clo = inputs[ElementsIDs.clo_input.value]
    v = inputs[ElementsIDs.v_input.value]
    rh = inputs[ElementsIDs.rh_input.value]

    return met, clo, tr, t_db, v, rh


def compare_get_inputs(inputs):
    met_2 = inputs[ElementsIDs.met_input_input2.value]
    clo_2 = inputs[ElementsIDs.clo_input_input2.value]
    tr_2 = inputs[ElementsIDs.t_r_input_input2.value]
    t_db_2 = inputs[ElementsIDs.t_db_input_input2.value]
    v_2 = inputs[ElementsIDs.v_input_input2.value]
    rh_2 = inputs[ElementsIDs.rh_input_input2.value]

    return met_2, clo_2, tr_2, t_db_2, v_2, rh_2


def adaptive_chart(
    inputs: dict = None,
    model: str = "iso",
    units: str = "SI",
):
    traces = []

    if units == UnitSystem.IP.value:
        x_values = np.array([50, 92.3]) if model == "iso" else np.array([50, 92.3])
    else:
        x_values = np.array([10, 30]) if model == "iso" else np.array([10, 33.5])

    if model == "iso":
        adaptive_func = adaptive_en
    else:
        adaptive_func = adaptive_ashrae

    results_min = adaptive_func(
        tdb=inputs[ElementsIDs.t_db_input.value],
        tr=inputs[ElementsIDs.t_r_input.value],
        t_running_mean=x_values[0],
        v=inputs[ElementsIDs.v_input.value],
        units=units,
    )
    results_max = adaptive_func(
        tdb=inputs[ElementsIDs.t_db_input.value],
        tr=inputs[ElementsIDs.t_r_input.value],
        t_running_mean=x_values[1],
        v=inputs[ElementsIDs.v_input.value],
        units=units,
    )

    if model == "iso":
        categories = [
            ("cat_iii", "Category III", "rgba(144, 238, 144, 0.3)"),
            ("cat_ii", "Category II", "rgba(34, 139, 34, 0.5)"),
            ("cat_i", "Category I", "rgba(0, 100, 0, 0.7)"),
        ]
    else:
        categories = [
            ("80", "80% Acceptability", "rgba(0, 100, 200, 0.2)"),
            ("90", "90% Acceptability", "rgba(0, 100, 200, 0.4)"),
        ]

    for cat, name, color in categories:
        y_values_up = [
            results_min[f"tmp_cmf_{cat}_up"],
            results_max[f"tmp_cmf_{cat}_up"],
        ]
        y_values_low = [
            results_min[f"tmp_cmf_{cat}_low"],
            results_max[f"tmp_cmf_{cat}_low"],
        ]

        traces.append(
            go.Scatter(
                x=np.concatenate([x_values, x_values[::-1]]),
                y=np.concatenate([y_values_up, y_values_low[::-1]]),
                fill="toself",
                fillcolor=color,
                line=dict(color="rgba(0,0,0,0)", shape="linear"),
                name=name,
                mode="lines",
            )
        )

    x = inputs[ElementsIDs.t_rm_input.value]
    y = t_o(
        tdb=inputs[ElementsIDs.t_db_input.value],
        tr=inputs[ElementsIDs.t_r_input.value],
        v=inputs[ElementsIDs.v_input.value],
    )
    red_point = [x, y]
    traces.append(
        go.Scatter(
            x=[red_point[0]],
            y=[red_point[1]],
            mode="markers",
            marker=dict(
                color="red",
                size=6,
            ),
            name="current input",
            showlegend=False,
        )
    )

    layout = go.Layout(
        xaxis=dict(
            title=(
                "Outdoor Running Mean Temperature [°C]"
                if units == UnitSystem.SI.value
                else "Prevailing Mean Outdoor Temperature [°F]"
            ),
            range=[10, 30] if model == "iso" else [10, 33.5],
            dtick=2 if units == UnitSystem.SI.value else 5,
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1,
            ticks="outside",
            ticklen=5,
            showline=True,
            linewidth=1.5,
            linecolor="black",
        ),
        yaxis=dict(
            title=(
                "Operative Temperature [°C]"
                if units == UnitSystem.SI.value
                else "Operative Temperature [°F]"
            ),
            range=[14, 36] if units == UnitSystem.SI.value else [60, 104],
            dtick=2 if units == UnitSystem.SI.value else 5,
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1,
            ticks="outside",
            ticklen=5,
            showline=True,
            linewidth=1.5,
            linecolor="black",
        ),
        legend=dict(x=0.8, y=1),
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=40),
    )

    fig = go.Figure(data=traces, layout=layout)

    if units == UnitSystem.IP.value:
        fig.update_layout(
            xaxis=dict(
                range=(
                    [50, 92.3]
                    if model == "iso"
                    else [
                        UnitConverter.celsius_to_fahrenheit(10),
                        UnitConverter.celsius_to_fahrenheit(33.5),
                    ]
                ),
            ),
        )

    return fig


def t_rh_pmv(
    inputs: dict = None,
    model: str = "iso",
    function_selection: str = Functionalities.Default,
    units: str = "SI",
):
    results = []
    if model == "iso":
        pmv_limits = [-0.7, -0.5, -0.2, 0.2, 0.5, 0.7]
        colors = [
            "rgba(168,204,162,0.9)",
            "rgba(114,174,106,0.9)",
            "rgba(78,156,71,0.9)",
            "rgba(114,174,106,0.9)",
            "rgba(168,204,162,0.9)",
        ]
    else:  # ASHRAE
        pmv_limits = [-0.5, 0.5]
        colors = ["rgba(59, 189, 237, 0.7)"]

    met, clo, tr, t_db, v, rh = get_inputs(inputs)
    clo_d = clo_dynamic(clo, met)
    vr = v_relative(v, met)

    def calculate_pmv_results(tr, vr, met, clo):
        results = []
        for pmv_limit in pmv_limits:
            for rh in np.arange(0, 110, 10):

                def function(x):
                    return (
                        pmv(
                            x,
                            tr=tr,
                            vr=vr,
                            rh=rh,
                            met=met,
                            clo=clo,
                            wme=0,
                            standard=model,
                            units=units,
                            limit_inputs=False,
                        )
                        - pmv_limit
                    )

                try:
                    temp = optimize.brentq(function, 10, 120)
                    results.append(
                        {
                            "rh": rh,
                            "temp": temp,
                            "pmv_limit": pmv_limit,
                        }
                    )
                except ValueError:
                    continue
        return pd.DataFrame(results)

    df = calculate_pmv_results(
        tr=tr,
        vr=vr,
        met=met,
        clo=clo_d,
    )

    fig = go.Figure()

    for i in range(len(pmv_limits) - 1):
        t1 = df[df["pmv_limit"] == pmv_limits[i]]
        t2 = df[df["pmv_limit"] == pmv_limits[i + 1]]
        fig.add_trace(
            go.Scatter(
                x=t1["temp"],
                y=t1["rh"],
                fill=None,
                mode="lines",
                line=dict(color=colors[i]),
                name=f"{model} Lower Limit",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t2["temp"],
                y=t2["rh"],
                fill="tonexty",
                mode="lines",
                fillcolor=colors[i],
                line=dict(color=colors[i]),
                name=f"{model} Upper Limit",
                hoverinfo="skip",
            )
        )

    # Add scatter point for the current input
    fig.add_trace(
        go.Scatter(
            x=[t_db],
            y=[rh],
            mode="markers",
            marker=dict(color="red", size=8),
            name="Current Input",
            # hoverinfo="skip",
        )
    )

    if model == "ashrae" and function_selection == Functionalities.Compare.value:
        met_2, clo_2, tr_2, t_db_2, v_2, rh_2 = compare_get_inputs(inputs)
        clo_d_compare = clo_dynamic(clo_2, met_2)
        vr_compare = v_relative(v_2, met_2)

        df_compare = calculate_pmv_results(
            tr_2,
            vr_compare,
            met_2,
            clo_d_compare,
        )
        t1_compare = df_compare[df_compare["pmv_limit"] == pmv_limits[0]]
        t2_compare = df_compare[df_compare["pmv_limit"] == pmv_limits[1]]
        fig.add_trace(
            go.Scatter(
                x=t1_compare["temp"],
                y=t1_compare["rh"],
                fill=None,
                mode="lines",
                line=dict(color="rgba(30,70,100,0.5)"),
                name=f"{model} Compare Lower Limit",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t2_compare["temp"],
                y=t2_compare["rh"],
                fill="tonexty",
                mode="lines",
                fillcolor="rgba(30,70,100,0.5)",
                line=dict(color="rgba(30,70,100,0.5)"),
                name=f"{model} Compare Upper Limit",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[t_db_2],
                y=[rh_2],
                mode="markers",
                marker=dict(color="blue", size=8),
                name="Compare Input",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        yaxis=dict(title="Relative Humidity [%]", range=[0, 100], dtick=10),
        xaxis=dict(title="Dry-bulb Temperature (°C)", range=[10, 36], dtick=2),
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=40),
    )

    if units == UnitSystem.IP.value:
        fig.update_layout(
            xaxis=dict(title="Dry-bulb Temperature [°F]", range=[50, 100], dtick=5),
        )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.2)")

    return fig

