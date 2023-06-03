from trytune.services.scheduler import SchedulerInner
from typing import Any, Dict


class FifoScheduler(SchedulerInner):
    """
    A scheduler that schedules the requests in a FIFO manner.

    Attributes:
        config (dict): The configuration of the scheduler.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        pass

    async def inference(self, model: str, request: Any) -> Any:
        # TODO: Spawn a new async task to handle the request (directry call the triton server)
        # TODO: When testing, use a mock triton server
        raise NotImplementedError

    async def start(self) -> Any:
        raise NotImplementedError

    async def stop(self) -> Any:
        raise NotImplementedError

    def metadata(self) -> Any:
        pass
