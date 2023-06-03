from pydantic import BaseModel
from typing import Optional, Dict, Any


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
    config: Optional[Dict[str, Any]] = None
