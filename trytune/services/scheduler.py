from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List
from trytune.schemas.common import InferSchema, DataSchema
from trytune.services.schedulers import fifo, common

# FIXME: Avoid using singleton pattern
class Scheduler:
    def __init__(self) -> None:
        self.inner: Optional[common.SchedulerInner] = None

    async def set_inner(self, scheduler: str, config: Dict[str, Any]) -> None:
        if scheduler == "fifo":
            self.inner = fifo.FifoScheduler(config)
            await self.inner.start()
        else:
            raise Exception(f"Scheduler {scheduler} is not supported")

    def get_metadata(self) -> Any:
        if self.inner is None:
            raise Exception("Scheduler inner is not set")
        return self.inner.metadata()

    async def delete_inner(self) -> None:
        if self.inner is None:
            raise Exception("Scheduler inner is not set")
        await self.inner.stop()
        self.inner = None

    # Return the output of the request as a list of DataSchema
    async def infer(self, schema: InferSchema) -> List[DataSchema]:
        if self.inner is None:
            raise Exception("Scheduler inner is not set")
        return await self.inner.infer(schema)
