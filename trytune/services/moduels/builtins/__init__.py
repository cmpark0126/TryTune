from trytune.services.moduels.builtins.crop import Crop
from trytune.services.moduels.builtins.fasterrcnn_resnet50_fpn import (
    FasterRCNN_ResNet50_FPN,
)
from trytune.services.moduels.builtins.resnet50_from_torch_hub import (
    Resnet50FromTorchHub,
)

__all__ = [
    "Crop",
    "FasterRCNN_ResNet50_FPN",
    "Resnet50FromTorchHub",
]
