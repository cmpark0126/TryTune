from typing import Any, Dict
from trytune.schemas.common import DataSchema
from trytune.services.moduels import BuiltinModule


class FasterRCNN_ResNet50_FPN(BuiltinModule):
    async def initialize(self, args: Dict[str, Any]) -> None:
        pass

    async def execute(self, requests: Any) -> Dict[str, DataSchema]:
        raise NotImplementedError

    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError


class NMS(BuiltinModule):
    """
    Non-maximum suppression
    """

    async def initialize(self, args: Dict[str, Any]) -> None:
        self.threshold = args["threshold"]

    async def execute(self, requests: Any) -> Dict[str, DataSchema]:
        # If request has threshold, use it, otherwise use self.threshold
        raise NotImplementedError

    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError


class Crop(BuiltinModule):
    async def initialize(self, args: Dict[str, Any]) -> None:
        pass

    async def execute(self, requests: Any) -> Dict[str, DataSchema]:
        raise NotImplementedError

    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError
