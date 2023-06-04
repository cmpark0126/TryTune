from typing import Any, Dict, List
from trytune.schemas.common import InferSchema, DataSchema
import trytune.services.schedulers.common as common


class FifoScheduler(common.SchedulerInner):
    """
    A scheduler that schedules the requests in a FIFO manner.

    Attributes:
        config (dict): The configuration of the scheduler.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        pass

    async def infer(self, schema: InferSchema) -> List[DataSchema]:
        raise NotImplementedError

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def metadata(self) -> Dict[str, Any]:
        return {"name": "fifo", "config": self.config}
