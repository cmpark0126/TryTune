import pytest
from pydantic import ValidationError
from trytune.schemas.pipelines import PipelineAddSchema


def test_pipeline_add_schema() -> None:
    valid_data = {
        "name": "pipe1",
        "tensors": {
            "inputs": [{"name": "pinput__0"}],
            "outputs": [{"name": "poutput__0"}],
            "interms": [{"name": "pinterm__0"}],
        },
        # pinput__0    -> [classifier] -> pinterm__0          -> [selector] -> poutput__0
        # input_tensor -> [stage]      -> intermediate_tensor -> [stage]    -> output_tensor
        "stages": [
            {
                "name": "classifier",
                "model": "resnet50",
                "inputs": [{"src": "input__0", "tgt": "pinput__0"}],
                "outputs": [{"src": "output__0", "tgt": "pinterm__0"}],
            },
            {
                "name": "selector",
                "model": "top_five",
                "inputs": [{"src": "input__0", "tgt": "pinterm__0"}],
                "outputs": [{"src": "output__0", "tgt": "poutput__0"}],
            },
        ],
    }

    # Test valid data
    try:
        pipeline = PipelineAddSchema(**valid_data)
    except ValidationError as e:
        assert False, f"Failed to create PipelineAddSchema instance with valid data: {e}"

    # Access schema fields
    assert pipeline.name == "pipe1"
    assert len(pipeline.tensors.inputs) == 1
    assert pipeline.tensors.inputs[0].name == "pinput__0"
    assert len(pipeline.tensors.outputs) == 1
    assert pipeline.tensors.outputs[0].name == "poutput__0"
    assert len(pipeline.tensors.interms) == 1
    assert pipeline.tensors.interms[0].name == "pinterm__0"
    assert len(pipeline.stages) == 2
    assert pipeline.stages[0].name == "classifier"
    assert pipeline.stages[0].model == "resnet50"
    assert len(pipeline.stages[0].inputs) == 1
    assert pipeline.stages[0].inputs[0].src == "input__0"
    assert pipeline.stages[0].inputs[0].tgt == "pinput__0"
    assert len(pipeline.stages[0].outputs) == 1
    assert pipeline.stages[0].outputs[0].src == "output__0"
    assert pipeline.stages[0].outputs[0].tgt == "pinterm__0"

    # Test missing required field
    invalid_data = {
        "name": "pipe1",
        "tensors": {
            "inputs": [{"name": "pinput__0"}],
            "outputs": [{"name": "poutput__0"}],
            "interms": [{"name": "pinterm__0"}],
        },
        "stages": [
            {
                "name": "classifier",
                "model": "resnet50",
                # invalid field
                "inputs": [{"src": "input__0"}],
                "outputs": [{"src": "output__0", "tgt": "pinterm__0"}],
            },
            {
                "name": "selector",
                "model": "top_five",
                "inputs": [{"src": "input__0", "tgt": "pinterm__0"}],
                "outputs": [{"src": "output__0", "tgt": "poutput__0"}],
            },
        ],
    }

    with pytest.raises(ValidationError):
        PipelineAddSchema(**invalid_data)

    # Test missing required field
    invalid_data = {
        "name": "pipe1",
        "tensors": {
            "inputs": [{"name": "pinput__0"}],
            "outputs": [{"name": "poutput__0"}],
            # invalid field
            "tensors": [{"name": "pinterm__0"}],
        },
        "stages": [
            {
                "name": "classifier",
                "model": "resnet50",
                "inputs": [{"src": "input__0", "tgt": "pinput__0"}],
                "outputs": [{"src": "output__0", "tgt": "pinterm__0"}],
            },
            {
                "name": "selector",
                "model": "top_five",
                "inputs": [{"src": "input__0", "tgt": "pinterm__0"}],
                "outputs": [{"src": "output__0", "tgt": "poutput__0"}],
            },
        ],
    }

    with pytest.raises(ValidationError):
        PipelineAddSchema(**invalid_data)
