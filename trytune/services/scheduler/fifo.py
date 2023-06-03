from trytune.services.scheduler import SchedulerInner
from typing import Any, Dict


class FifoScheduler(SchedulerInner):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        pass

    async def inference(self, model: str, request: Any) -> Any:
        # TODO: Spawn a new async task to handle the request (directry call the triton server)
        # TODO: When testing, use a mock triton server
        raise NotImplementedError
