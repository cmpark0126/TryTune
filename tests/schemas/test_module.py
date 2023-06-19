from pydantic import ValidationError
import pytest

from trytune.schemas.module import AddModuleSchema, ModuleTypeSchema


def test_add_module_schema() -> None:
    valid_data = {
        "name": "test_module",
        "type": "triton",
        "urls": {
            "g4dn.xlarge": "http://example.com/g4dn",
        },
    }
    # Test valid data
    add_module = AddModuleSchema(**valid_data)
    assert add_module.name == "test_module"
    assert add_module.type == ModuleTypeSchema.TRITON
    assert add_module.urls is not None
    assert add_module.urls["g4dn.xlarge"] == "http://example.com/g4dn"

    # Test invalid url
    invalid_data = valid_data = {
        "name": "test_module",
        "type": "triton",
        "urls": {
            "g4dn.xlarge": "not_a_valid_url",
        },
    }
    with pytest.raises(ValidationError):
        AddModuleSchema(**invalid_data)

    # Test invalid builtin_args
    invalid_data = {
        "name": "test_module",
        "type": "triton",
        "urls": {},
    }
    with pytest.raises(ValidationError):
        AddModuleSchema(**invalid_data)

    # Test valid builtin
    valid_data = {
        "name": "test_module",
        "type": "builtin",
        "builtin_args": {
            "arg1": "value1",
        },
    }
    add_module = AddModuleSchema(**valid_data)
    assert add_module.name == "test_module"
    assert add_module.type == ModuleTypeSchema.BUILTIN
    assert add_module.builtin_args is not None
    assert add_module.builtin_args["arg1"] == "value1"

    # Test invalid builtin_args
    invalid_data = {"name": "test_module", "type": "builtin", "builtin_args": {}}
    with pytest.raises(ValidationError):
        AddModuleSchema(**invalid_data)
