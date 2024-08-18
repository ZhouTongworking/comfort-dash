from utils.my_config_file import ModelInputs,ModelInputsInfo, ModelInputs2
import dash_mantine_components as dmc
from dataclasses import fields


def input_environmental_personal(selected_model):
    inputs = []

    model_inputs = ModelInputs()
    if selected_model =='EN - 16798':
        print("EN")
        model_inputs = ModelInputs2()
    elif selected_model == 'PMV - ASHRAE 55':
        model_inputs = ModelInputs()
        print("PMV")

    for var_name, values in dict(model_inputs).items():
        input_filed = dmc.NumberInput(
            label=values.name,
            description=f"From {values.min} to {values.max}",
            value=values.value,
            min=values.min,
            max=values.max,
            step=values.step,
        )
        inputs.append(input_filed)

    return dmc.Paper(
    children=[
        dmc.Text("Inputs", mb="xs", fw=700),
        dmc.Stack(inputs, gap="xs")
    ],
    shadow="md",
    p="md"
)
