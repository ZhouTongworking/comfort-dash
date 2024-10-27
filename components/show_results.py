import dash_mantine_components as dmc
from pythermalcomfort.models import pmv_ppd, adaptive_ashrae
from pythermalcomfort.utilities import v_relative, clo_dynamic, mapping
from pythermalcomfort.models import adaptive_en, set_tmp, pmv_ppd, cooling_effect
from pythermalcomfort.psychrometrics import t_o
from utils.my_config_file import (
    Models,
    UnitSystem,
    ElementsIDs,
    Functionalities,
    CompareInputColor,
    ComfortLevel,
    Charts,
)


def create_text_component(text, color=None, center=True):
    text_component = dmc.Text(text, style={"color": color} if color else {})
    return dmc.Center(text_component) if center else text_component


def format_value_with_unit(value, units, has_decimals=True):
    temp_unit = "°F" if units == UnitSystem.IP.value else "°C"
    format_str = f"{value:.1f}" if has_decimals else str(value)
    return f"{format_str} {temp_unit}"


def create_pmv_display_items(
    pmv_results, set_temperature, comfort_category=None, units=None
):
    base_items = [
        ("PMV", f"{pmv_results['pmv']:.2f}"),
        ("PPD", f"{pmv_results['ppd']:.1f} %"),
    ]

    if units:
        base_items.append(("SET", format_value_with_unit(set_temperature, units)))

    if comfort_category is not None:
        category_label = (
            "Sensation"
            if comfort_category
            in [
                "Cold",
                "Cool",
                "Slightly Cool",
                "Neutral",
                "Slightly Warm",
                "Warm",
                "Hot",
            ]
            else "Category"
        )
        base_items.append((category_label, comfort_category))

    return base_items


def calculate_pmv_results(inputs, is_input2=False, units=None, standard="ISO"):
    suffix = "_input2" if is_input2 else ""

    r_pmv = pmv_ppd(
        tdb=inputs[getattr(ElementsIDs, f"t_db_input{suffix}").value],
        tr=inputs[getattr(ElementsIDs, f"t_r_input{suffix}").value],
        vr=v_relative(
            v=inputs[getattr(ElementsIDs, f"v_input{suffix}").value],
            met=inputs[getattr(ElementsIDs, f"met_input{suffix}").value],
        ),
        rh=inputs[getattr(ElementsIDs, f"rh_input{suffix}").value],
        met=inputs[getattr(ElementsIDs, f"met_input{suffix}").value],
        clo=clo_dynamic(
            clo=inputs[getattr(ElementsIDs, f"clo_input{suffix}").value],
            met=inputs[getattr(ElementsIDs, f"met_input{suffix}").value],
        ),
        wme=0,
        limit_inputs=True,
        units=units,
        standard=standard,
    )

    r_set = set_tmp(
        tdb=inputs[getattr(ElementsIDs, f"t_db_input{suffix}").value],
        tr=inputs[getattr(ElementsIDs, f"t_r_input{suffix}").value],
        v=v_relative(
            v=inputs[getattr(ElementsIDs, f"v_input{suffix}").value],
            met=inputs[getattr(ElementsIDs, f"met_input{suffix}").value],
        ),
        rh=inputs[getattr(ElementsIDs, f"rh_input{suffix}").value],
        met=inputs[getattr(ElementsIDs, f"met_input{suffix}").value],
        clo=clo_dynamic(
            clo=inputs[getattr(ElementsIDs, f"clo_input{suffix}").value],
            met=inputs[getattr(ElementsIDs, f"met_input{suffix}").value],
        ),
        wme=0,
        limit_inputs=True,
        units=units,
        standard=standard,
    )

    r_cooling = cooling_effect(
        tdb=inputs[getattr(ElementsIDs, f"t_db_input{suffix}").value],
        tr=inputs[getattr(ElementsIDs, f"t_r_input{suffix}").value],
        vr=v_relative(
            v=inputs[getattr(ElementsIDs, f"v_input{suffix}").value],
            met=inputs[getattr(ElementsIDs, f"met_input{suffix}").value],
        ),
        rh=inputs[getattr(ElementsIDs, f"rh_input{suffix}").value],
        met=inputs[getattr(ElementsIDs, f"met_input{suffix}").value],
        clo=clo_dynamic(
            clo=inputs[getattr(ElementsIDs, f"clo_input{suffix}").value],
            met=inputs[getattr(ElementsIDs, f"met_input{suffix}").value],
        ),
        wme=0,
        units=units,
    )

    return r_pmv, r_set, r_cooling


def check_compliance(pmv_value, is_ashrae):
    if is_ashrae:
        is_compliant = -0.5 <= pmv_value <= 0.5
        text = "✔" if is_compliant else "✘"
    else:
        is_compliant = -0.7 <= pmv_value <= 0.7
        text = "✔" if is_compliant else "✘"
    color = "green" if is_compliant else "red"
    return text, color


def create_compare_first_col(units):
    base_titles = [
        "Compliance",
        "PMV",
        "PPD",
        "Sensation",
        "SET",
    ]

    if units == UnitSystem.IP.value:
        base_titles.extend(["Dry-bulb temp at still air", "Cooling effect"])

    return [
        dmc.Stack(
            children=[dmc.Center(dmc.Text(title)) for title in base_titles],
            gap=5,
            style={"textAlign": "left", "width": "100%"},
        )
    ]


def get_comfort_category(pmv_value, model):
    if model == Models.PMV_ashrae.name:
        return mapping(
            pmv_value,
            {
                -2.5: "Cold",
                -1.5: "Cool",
                -0.5: "Slightly Cool",
                0.5: "Neutral",
                1.5: "Slightly Warm",
                2.5: "Warm",
                10: "Hot",
            },
        )
    else:
        return mapping(
            abs(pmv_value), {0.2: "I", 0.5: "II", 0.7: "III", float("inf"): "IV"}
        )


def create_default_result(pmv_results, set_temperature, comfort_category, model, units):
    is_ashrae = model == Models.PMV_ashrae.name
    compliance_text = "✔  Complies with " + (
        "ASHRAE Standard 55-2023" if is_ashrae else "EN-16798"
    )
    if (is_ashrae and not (-0.5 <= pmv_results["pmv"] <= 0.5)) or (
        not is_ashrae and not (-0.7 <= pmv_results["pmv"] <= 0.7)
    ):
        compliance_text = "✘  Does not comply with " + (
            "ASHRAE Standard 55-2023" if is_ashrae else "EN-16798"
        )
        compliance_color = "red"
    else:
        compliance_color = "green"

    standard_checker = dmc.Text(
        compliance_text,
        c=compliance_color,
        ta="center",
        size="md",
        style={"width": "100%"},
    )

    grid_children = [
        create_text_component(f"{label}: {value}")
        for label, value in create_pmv_display_items(
            pmv_results, set_temperature, comfort_category, units if is_ashrae else None
        )
    ]

    results = [
        standard_checker,
        dmc.SimpleGrid(
            cols=3 if not is_ashrae else 2,
            spacing="xs",
            verticalSpacing="xs",
            children=grid_children,
        ),
    ]

    for child in results[1].children:
        if isinstance(child, dmc.Center) and isinstance(child.children, dmc.Text):
            child.children.style = {"color": CompareInputColor.InputColor1.value}

    return results


def create_result_stack(
    pmv_results, set_temperature, cooling_result, t_db, units, color
):
    comfort_category = mapping(
        pmv_results["pmv"],
        {
            -2.5: "Cold",
            -1.5: "Cool",
            -0.5: "Slightly Cool",
            0.5: "Neutral",
            1.5: "Slightly Warm",
            2.5: "Warm",
            10: "Hot",
        },
    )

    compliance_text, compliance_color = check_compliance(pmv_results["pmv"], True)

    children = [create_text_component(compliance_text, compliance_color)]

    display_items = create_pmv_display_items(
        pmv_results, set_temperature, comfort_category, units
    )
    children.extend(create_text_component(value) for _, value in display_items)

    if units == UnitSystem.IP.value:
        children.extend(
            [
                create_text_component(f"{t_db}"),
                create_text_component(f"{cooling_result:.1f}"),
            ]
        )

    stack = dmc.Stack(
        children=children,
        gap=5,
        style={"textAlign": "center", "width": "100%"},
    )

    for child in stack.children[1:]:
        if isinstance(child, dmc.Center) and isinstance(child.children, dmc.Text):
            child.children.style = {"color": color}

    return stack


def display_results(inputs: dict):
    selected_model: str = inputs[ElementsIDs.MODEL_SELECTION.value]
    units: str = inputs[ElementsIDs.UNIT_TOGGLE.value]
    results = []

    if selected_model in [Models.PMV_EN.name, Models.PMV_ashrae.name]:
        standard = "ashrae" if selected_model == Models.PMV_ashrae.name else "ISO"

        if (
            inputs[ElementsIDs.functionality_selection.value]
            == Functionalities.Compare.value
            and selected_model == Models.PMV_ashrae.name
        ):

            results_title = create_compare_first_col(units)
            r_pmv, r_set, r_cooling = calculate_pmv_results(
                inputs, False, units, standard
            )
            r_pmv2, r_set2, r_cooling2 = calculate_pmv_results(
                inputs, True, units, standard
            )

            results = create_result_stack(
                r_pmv,
                r_set,
                r_cooling,
                inputs[ElementsIDs.t_db_input.value],
                units,
                CompareInputColor.InputColor1.value,
            )

            results2 = create_result_stack(
                r_pmv2,
                r_set2,
                r_cooling2,
                inputs[ElementsIDs.t_db_input_input2.value],
                units,
                CompareInputColor.InputColor2.value,
            )

            return dmc.Grid(
                children=[
                    dmc.Stack(
                        children=results_title,
                        style={"flex": "1", "display": "inline-block"},
                    ),
                    dmc.Stack(
                        children=[results],
                        style={"flex": "1", "display": "inline-block"},
                    ),
                    dmc.Stack(
                        children=[results2],
                        style={"flex": "1", "display": "inline-block"},
                    ),
                ],
                style={"display": "flex"},
            )
        else:
            r_pmv, r_set, r_cooling = calculate_pmv_results(
                inputs, False, units, standard
            )
            comfort_category = get_comfort_category(r_pmv["pmv"], selected_model)
            return dmc.Stack(
                children=create_default_result(
                    r_pmv, r_set, comfort_category, selected_model, units
                ),
                gap=0,
                align="stretch",
            )

    elif selected_model == Models.Adaptive_EN.name:
        results = gain_adaptive_en_hover_text(
            tdb=inputs[ElementsIDs.t_db_input.value],
            tr=inputs[ElementsIDs.t_r_input.value],
            trm=inputs[ElementsIDs.t_rm_input.value],
            v=inputs[ElementsIDs.v_input.value],
            units=units,
        )
    elif selected_model == Models.Adaptive_ASHRAE.name:
        results = gain_adaptive_ashrae_hover_text(
            tdb=inputs[ElementsIDs.t_db_input.value],
            tr=inputs[ElementsIDs.t_r_input.value],
            trm=inputs[ElementsIDs.t_rm_input.value],
            v=inputs[ElementsIDs.v_input.value],
            units=units,
        )

    if selected_model == Models.PMV_ashrae.name:
        if (
            inputs[ElementsIDs.chart_selected.value] == Charts.set_outputs.value.name
            or inputs[ElementsIDs.chart_selected.value]
            == Charts.thl_psychrometric.value.name
        ):
            return None

    return dmc.Stack(
        children=results,
        gap=0,
        align="stretch",
    )


def gain_adaptive_en_hover_text(tdb, tr, trm, v, units):
    if tdb is None or tr is None or trm is None or v is None:
        return "None"

    result = adaptive_en(tdb=tdb, tr=tr, t_running_mean=trm, v=v, units=units)
    y = t_o(tdb=tdb, tr=tr, v=v)
    if y > result["tmp_cmf_cat_iii_up"] or y < result["tmp_cmf_cat_iii_low"]:
        compliance_text = "✘ Does not comply with EN 16798"
        compliance_color = "red"
    else:
        compliance_text = "✔ Complies with EN 16798"
        compliance_color = "green"

    if result["tmp_cmf_cat_i_low"] <= y <= result["tmp_cmf_cat_i_up"]:
        class3_bool = ComfortLevel.COMFORTABLE
        class2_bool = ComfortLevel.COMFORTABLE
        class1_bool = ComfortLevel.COMFORTABLE
    elif result["tmp_cmf_cat_i_up"] < y <= result["tmp_cmf_cat_ii_up"]:
        class3_bool = ComfortLevel.COMFORTABLE
        class2_bool = ComfortLevel.COMFORTABLE
        class1_bool = ComfortLevel.TOO_WARM
    elif result["tmp_cmf_cat_ii_up"] < y <= result["tmp_cmf_cat_iii_up"]:
        class3_bool = ComfortLevel.COMFORTABLE
        class2_bool = ComfortLevel.TOO_WARM
        class1_bool = ComfortLevel.TOO_WARM
    elif result["tmp_cmf_cat_iii_up"] < y:
        class3_bool = ComfortLevel.TOO_WARM
        class2_bool = ComfortLevel.TOO_WARM
        class1_bool = ComfortLevel.TOO_WARM
    elif result["tmp_cmf_cat_i_low"] > y >= result["tmp_cmf_cat_ii_low"]:
        class3_bool = ComfortLevel.COMFORTABLE
        class2_bool = ComfortLevel.COMFORTABLE
        class1_bool = ComfortLevel.TOO_COOL
    elif result["tmp_cmf_cat_ii_low"] > y >= result["tmp_cmf_cat_iii_low"]:
        class3_bool = ComfortLevel.COMFORTABLE
        class2_bool = ComfortLevel.TOO_COOL
        class1_bool = ComfortLevel.TOO_COOL
    elif result["tmp_cmf_cat_iii_low"] > y:
        class3_bool = ComfortLevel.TOO_COOL
        class2_bool = ComfortLevel.TOO_COOL
        class1_bool = ComfortLevel.TOO_COOL
    else:
        class3_bool = ComfortLevel.COMFORTABLE
        class2_bool = ComfortLevel.COMFORTABLE
        class1_bool = ComfortLevel.COMFORTABLE

    results = []
    temp_unit = "°F" if units == UnitSystem.IP.value else "°C"
    results.append(
        dmc.Text(
            compliance_text,
            c=compliance_color,
            ta="center",
            size="md",
            style={"width": "100%"},
        )
    )
    results.append(
        dmc.Center(
            dmc.Text(
                f"Class III acceptability limits = Operative temperature: {result['tmp_cmf_cat_iii_low']} to {result['tmp_cmf_cat_iii_up']} {temp_unit}"
            )
        )
    )
    results.append(
        dmc.Center(dmc.Text(f"{class3_bool.description}", fz="xs", c=class3_bool.color))
    )
    results.append(
        dmc.Center(
            dmc.Text(
                f"Class II acceptability limits = Operative temperature: {result['tmp_cmf_cat_ii_low']} to {result['tmp_cmf_cat_ii_up']} {temp_unit}"
            )
        )
    )
    results.append(
        dmc.Center(dmc.Text(f"{class2_bool.description}", fz="xs", c=class2_bool.color))
    )
    results.append(
        dmc.Center(
            dmc.Text(
                f"Class I acceptability limits = Operative temperature: {result['tmp_cmf_cat_i_low']} to {result['tmp_cmf_cat_i_up']} {temp_unit}"
            )
        )
    )
    results.append(
        dmc.Center(dmc.Text(f"{class1_bool.description}", fz="xs", c=class1_bool.color))
    )
    return results


def gain_adaptive_ashrae_hover_text(tdb, tr, trm, v, units):
    if tdb is None or tr is None or trm is None or v is None:
        return "None"

    result = adaptive_ashrae(tdb=tdb, tr=tr, t_running_mean=trm, v=v, units=units)
    y = t_o(tdb=tdb, tr=tr, v=v)

    if y > result["tmp_cmf_80_up"] or y < result["tmp_cmf_80_low"]:
        compliance_text = "✘ Does not comply with ASHRAE 55"
        compliance_color = "red"
    else:
        compliance_text = "✔ Complies with ASHRAE 55"
        compliance_color = "green"

    if result["tmp_cmf_90_low"] <= y <= result["tmp_cmf_90_up"]:
        class2_bool = ComfortLevel.COMFORTABLE
        class1_bool = ComfortLevel.COMFORTABLE
    elif result["tmp_cmf_90_up"] < y <= result["tmp_cmf_80_up"]:
        class2_bool = ComfortLevel.COMFORTABLE
        class1_bool = ComfortLevel.TOO_WARM
    elif result["tmp_cmf_90_up"] < y:
        class2_bool = ComfortLevel.TOO_WARM
        class1_bool = ComfortLevel.TOO_WARM
    elif result["tmp_cmf_90_low"] > y >= result["tmp_cmf_80_low"]:
        class2_bool = ComfortLevel.COMFORTABLE
        class1_bool = ComfortLevel.TOO_COOL
    elif result["tmp_cmf_80_low"] > y:
        class2_bool = ComfortLevel.TOO_COOL
        class1_bool = ComfortLevel.TOO_COOL
    else:
        class2_bool = ComfortLevel.COMFORTABLE
        class1_bool = ComfortLevel.COMFORTABLE

    results = []
    temp_unit = "°F" if units == UnitSystem.IP.value else "°C"
    results.append(
        dmc.Text(
            compliance_text,
            c=compliance_color,
            ta="center",
            size="md",
            style={"width": "100%"},
        )
    )
    results.append(
        dmc.Center(
            dmc.Text(
                f"80% acceptability limits = Operative temperature: {round(result.tmp_cmf_80_low,1)} to {round(result.tmp_cmf_80_up,1)} {temp_unit}"
            )
        )
    )
    results.append(
        dmc.Center(dmc.Text(f"{class2_bool.description}", fz="xs", c=class2_bool.color))
    )
    results.append(
        dmc.Center(
            dmc.Text(
                f"90% acceptability limits = Operative temperature: {round(result.tmp_cmf_90_low,1)} to {round(result.tmp_cmf_90_up,1)} {temp_unit}"
            )
        )
    )
    results.append(
        dmc.Center(dmc.Text(f"{class1_bool.description}", fz="xs", c=class1_bool.color))
    )
    return results
