from trytune.services.scheduler import SchedulerInner
from trytune.schemas.common import InferSchema, DataSchema
from typing import Any, Dict, List


class FifoScheduler(SchedulerInner):
    """
    A scheduler that schedules the requests in a FIFO manner.

    Attributes:
        config (dict): The configuration of the scheduler.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        pass

    async def infer(self, schema: InferSchema) -> List[DataSchema]:
        # TODO: Spawn a new async task to handle the request (directry call the triton server)
        # TODO: When testing, use a mock triton server
        raise NotImplementedError

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def metadata(self) -> Any:
        pass
