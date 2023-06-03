from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
from trytune.services.scheduler.fifo import FifoScheduler


class SchedulerInner(ABC):
    @abstractmethod
    async def inference(self, model: str, request: Any) -> Any:
        # TODO: we can use channel to send requests to the scheduler core
        # TODO: maybe we can use a queue to store the outputs of requests
        pass


# FIXME: Avoid using singleton pattern
class Scheduler:
    def __init__(self) -> None:
        self.inner: Optional[SchedulerInner] = None

    def set_inner(self, scheduler: str, config: Dict[str, Any]) -> None:
        if scheduler == "fifo":
            self.inner = FifoScheduler(config)
        else:
            raise Exception(f"Scheduler {scheduler} is not supported")

    async def send_request(self, model: str, request: Any) -> Any:
        if self.inner is None:
            raise Exception("Scheduler inner is not set")
        return await self.inner.inference(model, request)
