import pytest
from pydantic import ValidationError
from trytune.schemas.common import InferSchema


def test_infer_schema() -> None:
    valid_data = {
        "target": "pipe1",
        "inputs": [
            {"name": "i1", "data": [1, 2, 3]},
            {"name": "i2", "data": [4, 5, 6]},
        ],
    }

    # Test valid data
    try:
        infer = InferSchema(**valid_data)
    except ValidationError as e:
        assert False, f"Failed to create InferSchema instance with valid data: {e}"

    # Access schema fields
    assert infer.target == "pipe1"
    assert infer.inputs[0].name == "i1"
    assert infer.inputs[0].data == [1, 2, 3]
    assert infer.inputs[1].name == "i2"
    assert infer.inputs[1].data == [4, 5, 6]

    # Test missing required field
    invalid_data = {
        "target": "pipe1",
        "inputs": [{"data": [1, 2, 3]}, {"name": "i2", "data": [4, 5]}],  # no name for first input
    }

    with pytest.raises(ValidationError):
        InferSchema(**invalid_data)

    invalid_data = {
        "target": "pipe1",
        "inputs": [
            {"name": "i2", "data": [1, 2, 3]},
            {"name": "i2"},
        ],  # no data for second input
    }

    with pytest.raises(ValidationError):
        InferSchema(**invalid_data)
