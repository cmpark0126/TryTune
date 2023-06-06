import pytest
from pydantic import ValidationError
from typing import Dict, Any
from trytune.schemas.module import AddModuleSchema


def test_add_module_schema() -> None:
    valid_data = {
        "name": "test_module",
        "urls": {
            "g4dn.xlarge": "eks.ingress.url/g4dn",
            "g5.xlarge": "eks.ingress.url/g5",
            "inf1.xlarge": "eks.ingress.url/inf1",
        },
    }

    # Test valid data
    try:
        add_module = AddModuleSchema(**valid_data)
    except ValidationError as e:
        assert False, f"Failed to create schema instance with valid data: {e}"

    # Access schema fields
    assert add_module.name == "test_module"
    assert add_module.urls["g4dn.xlarge"] == "eks.ingress.url/g4dn"
    assert add_module.urls["g5.xlarge"] == "eks.ingress.url/g5"
    assert add_module.urls["inf1.xlarge"] == "eks.ingress.url/inf1"

    # Test missing required field
    invalid_data: Dict[str, Any] = {
        "name": "test_module",
    }

    with pytest.raises(ValidationError):
        AddModuleSchema(**invalid_data)
