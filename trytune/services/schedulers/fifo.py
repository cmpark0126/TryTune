from typing import Any, Dict
from trytune.schemas.common import InferSchema
import trytune.services.schedulers.common as common
from trytune.services.moduels import modules


class FifoScheduler(common.SchedulerInner):
    """
    A scheduler that schedules the requests in a FIFO manner.

    Attributes:
        config (dict): The configuration of the scheduler.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        pass

    async def infer(self, schema: InferSchema) -> Dict[str, common.DataSchema]:
        module = modules.get(schema.target)

        metadata = module["metadata"]
        urls = module["metadata"]["urls"]
        assert len(urls) > 0
        instance_types = [instance_type for instance_type, _ in urls.items()]
        # TODO: use round robin to schedule the requests
        url = urls[instance_types[0]]

        return await common.infer_with_triton(url, metadata, schema.inputs)

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def metadata(self) -> Dict[str, Any]:
        return {"name": "fifo", "config": self.config}
