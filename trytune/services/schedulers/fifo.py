from typing import Any, Dict

import numpy as np

from trytune.schemas.module import ModuleTypeSchema
from trytune.services.moduels import modules
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

    async def infer(
        self, module_name: str, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        module = modules.get(module_name)
        metadata = module["metadata"]
        module_type = metadata["type"]
        if module_type == ModuleTypeSchema.TRITON:
            urls = metadata["urls"]
            assert len(urls) > 0
            instance_type = [instance_type for instance_type, _ in urls.items()][0]

            return await common.infer(module_name, inputs, instance_type=instance_type)
        elif module_type == ModuleTypeSchema.BUILTIN:
            return await common.infer(module_name, inputs)
        else:
            raise ValueError(f"Unknown module type {module_type}")

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def metadata(self) -> Dict[str, Any]:
        return {"name": "fifo", "config": self.config}
