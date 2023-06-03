import pytest
from pydantic import ValidationError
from typing import Dict, Any
from trytune.schemas.model import AddModelSchema


def test_add_model_schema() -> None:
    valid_data = {
        "name": "test_model",
        "urls": {
            "g4dn.xlarge": "eks.ingress.url/g4dn",
            "g5.xlarge": "eks.ingress.url/g5",
            "inf1.xlarge": "eks.ingress.url/inf1",
        },
    }

    # Test valid data
    try:
        add_model = AddModelSchema(**valid_data)
    except ValidationError as e:
        assert False, f"Failed to create schema instance with valid data: {e}"

    # Access schema fields
    assert add_model.name == "test_model"
    assert add_model.urls["g4dn.xlarge"] == "eks.ingress.url/g4dn"
    assert add_model.urls["g5.xlarge"] == "eks.ingress.url/g5"
    assert add_model.urls["inf1.xlarge"] == "eks.ingress.url/inf1"

    # Test missing required field
    invalid_data: Dict[str, Any] = {
        "name": "test_model",
    }

    with pytest.raises(ValidationError):
        AddModelSchema(**invalid_data)
