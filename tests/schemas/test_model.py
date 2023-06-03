import pytest
from pydantic import ValidationError
from trytune.schemas.model import ModelAddSchema


def test_model_add_schema():
    valid_data = {
        "linkes": {
            "g4dn.xlarge": "eks.ingress.url/g4dn",
            "g5.xlarge": "eks.ingress.url/g5",
            "inf1.xlarge": "eks.ingress.url/inf1",
        }
    }

    # Test valid data
    try:
        model_add = ModelAddSchema(**valid_data)
    except ValidationError as e:
        assert False, f"Failed to create schema instance with valid data: {e}"

    # Access schema fields
    assert model_add.linkes["g4dn.xlarge"] == "eks.ingress.url/g4dn"
    assert model_add.linkes["g5.xlarge"] == "eks.ingress.url/g5"
    assert model_add.linkes["inf1.xlarge"] == "eks.ingress.url/inf1"

    # Test missing required field
    invalid_data = {}

    with pytest.raises(ValidationError):
        ModelAddSchema(**invalid_data)
