from pydantic import ValidationError
import pytest

from trytune.schemas.scheduler import SetSchedulerSchema


def test_set_scheduler_schema() -> None:
    # Test SetSchedulerSchema with valid data
    valid_data = {
        "name": "fifo",
        "config": {"use_dynamic_batching": "true"},
    }

    try:
        set_scheduler = SetSchedulerSchema(**valid_data)
    except ValidationError as e:
        assert (
            False
        ), f"Failed to create SetSchedulerSchema instance with valid data: {e}"

    assert set_scheduler.name == "fifo"
    assert set_scheduler.config == {"use_dynamic_batching": "true"}

    # Test SetSchedulerSchema with invalid data
    invalid_data = {
        "name": "fifo",
        "config": "use_dynamic_batching",
    }

    with pytest.raises(ValueError):
        SetSchedulerSchema(**invalid_data)
