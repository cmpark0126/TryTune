import pytest
from pydantic import ValidationError
from trytune.schemas.scheudler import SetSchedulerSchema


def test_set_scheduler_schema() -> None:
    valid_data = {
        "name": "fifo",
        "config": {"use_dynamic_batching": "true"},
    }

    try:
        set_scheduler = SetSchedulerSchema(**valid_data)
    except ValidationError as e:
        assert False, f"Failed to create SetSchedulerSchema instance with valid data: {e}"

    assert set_scheduler.name == "fifo"
    assert set_scheduler.config == {"use_dynamic_batching": "true"}

    invalid_data = {
        "name": "fifo",
        "config": "use_dynamic_batching",
    }

    # 잘못된 입력에 대한 테스트
    with pytest.raises(ValueError):
        SetSchedulerSchema(**invalid_data)

    invalid_data_2 = {
        "name": "fifo",
        "config": {"use_dynamic_batching": "true"},
        "invalid": "invalid",
    }

    with pytest.raises(ValueError):
        SetSchedulerSchema(**invalid_data_2)
