from typing import Any, Dict
from trytune.schemas.common import DataSchema
from trytune.services.moduels import BuiltinModule
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights


class FasterRCNN_ResNet50_FPN(BuiltinModule):
    async def initialize(self, args: Dict[str, Any]) -> None:
        # Instantiate the PyTorch model
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights, progress=False)
        self.model.eval()
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
