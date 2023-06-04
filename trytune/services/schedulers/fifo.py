from typing import Any, Dict, List
from trytune.schemas.common import InferSchema, DataSchema
import trytune.services.schedulers.common as common

# from trytune.routers.models import models


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
        # model = models.get(schema.target)
        # clients = model["clients"]
        # assert len(clients) > 0
        # # TODO: use round robin to schedule the requests
        # client = clients[0]
        # metadata = model["metadata"]

        return [
            DataSchema(name="output__0", data=[0.0] * 1000),
            DataSchema(name="output__1", data=[0.0]),
        ]
        # return await common.infer_with_triton(client, metadata, schema.inputs)

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def metadata(self) -> Dict[str, Any]:
        return {"name": "fifo", "config": self.config}
