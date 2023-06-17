from pydantic import ValidationError
import pytest

from trytune.schemas.pipeline import AddPipelineSchema


def test_pipeline_add_schema() -> None:
    valid_data = {
        "name": "pipe1",
        "tensors": {
            "inputs": [{"name": "pinput__0"}],
            "outputs": [{"name": "poutput__0"}],
        },
        # pinput__0    -> [classifier] -> pinterm__0          -> [selector] -> poutput__0
        # input_tensor -> [stage]      -> intermediate_tensor -> [stage]    -> output_tensor
        "stages": [
            {
                "name": "classifier",
                "module": "resnet50",
                "inputs": {"input__0": "pinput__0"},
                "outputs": {"output__0": "pinterm__0"},
            },
            {
                "name": "selector",
                "module": "top_five",
                "inputs": {"input__0": "pinterm__0"},
                "outputs": {"output__0": "poutput__0"},
            },
        ],
    }

    # Test valid data
    try:
        add_pipeline = AddPipelineSchema(**valid_data)
    except ValidationError as e:
        assert False, f"Failed to create AddPipelineSchema instance with valid data: {e}"

    # Access schema fields
    assert add_pipeline.name == "pipe1"
    assert len(add_pipeline.tensors.inputs) == 1
    assert add_pipeline.tensors.inputs[0].name == "pinput__0"
    assert len(add_pipeline.tensors.outputs) == 1
    assert add_pipeline.tensors.outputs[0].name == "poutput__0"
    assert len(add_pipeline.stages) == 2
    assert add_pipeline.stages[0].name == "classifier"
    assert add_pipeline.stages[0].module == "resnet50"
    assert len(add_pipeline.stages[0].inputs) == 1
    assert add_pipeline.stages[0].inputs["input__0"] == "pinput__0"
    assert len(add_pipeline.stages[0].outputs) == 1
    assert add_pipeline.stages[0].outputs["output__0"] == "pinterm__0"

    # Test missing required field
    invalid_data = {
        "name": "pipe1",
        "tensors": {
            "inputs": [{"name": "pinput__0"}],
            "outputs": [{"name": "poutput__0"}],
        },
        "stages": [
            {
                "name": "classifier",
                "module": "resnet50",
                "inputs": {"input__0": "pinput__0"},
                # missing outputs
            },
            {
                "name": "selector",
                "module": "top_five",
                "inputs": {"input__0": "pinterm__0"},
                "outputs": {"output__0": "poutput__0"},
            },
        ],
    }

    with pytest.raises(ValidationError):
        AddPipelineSchema(**invalid_data)
