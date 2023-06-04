from abc import ABC, abstractmethod
from typing import Any, List
from trytune.schemas.common import InferSchema, DataSchema


class SchedulerInner(ABC):
    @abstractmethod
    async def infer(self, schema: InferSchema) -> List[DataSchema]:
        # TODO: we can use channel to send requests to the scheduler core
        # TODO: maybe we can use a queue to store the outputs of requests
        pass

    @abstractmethod
    async def start(self) -> Any:
        pass

    @abstractmethod
    async def stop(self) -> Any:
        pass

    @abstractmethod
    def metadata(self) -> Any:
        pass
