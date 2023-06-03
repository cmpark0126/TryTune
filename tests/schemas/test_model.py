import pytest
from pydantic import ValidationError
from typing import Dict, Any
from trytune.schemas.model import ModelAddSchema


def test_model_add_schema() -> None:
    valid_data = {
        "urls": {
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
    assert model_add.urls["g4dn.xlarge"] == "eks.ingress.url/g4dn"
    assert model_add.urls["g5.xlarge"] == "eks.ingress.url/g5"
    assert model_add.urls["inf1.xlarge"] == "eks.ingress.url/inf1"

    # Test missing required field
    invalid_data: Dict[str, Any] = {}

    with pytest.raises(ValidationError):
        ModelAddSchema(**invalid_data)
