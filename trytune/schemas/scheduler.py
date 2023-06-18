from typing import Any, Dict

from pydantic import BaseModel


class SetSchedulerSchema(BaseModel):
    """
    Schema for setting the scheduler

    Attributes:
        name: name of the scheduler
        config: configuration of the scheduler

    Example:
        {
            "name": "fifo",
            "config": {
                "use_dynamic_batching": "true",
            },
        },
    """

    name: str
    config: Dict[str, Any]
