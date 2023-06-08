from typing import Any, Dict

import numpy as np
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

from trytune.services.moduels.common import BuiltinModule


class FasterRCNN_ResNet50_FPN(BuiltinModule):
    async def initialize(self, args: Dict[str, Any]) -> None:
        # Instantiate the PyTorch model
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights, progress=False)
        self.model.eval()
        self.args = args
        pass

    async def execute(self, requests: Any) -> Any:
        inputs = requests["inputs"]
        batch_image = torch.from_numpy(inputs["BATCH_IMAGE"])

        # NOTE: batch_image must be B x C x H x W shapes
        preds = self.model(batch_image)
        # preds: [{"boxes": <torch.Size([41, 4])>, "labels": torch.Size([41]), "scores": torch.Size([41])}, ...]
        batch_boxes = []
        batch_labels = []
        batch_scores = []
        for pred in preds:
            # pred: {"boxes": <torch.Size([41, 4])>, "labels": torch.Size([41]), "scores": torch.Size([41])}
            batch_boxes.append(pred["boxes"].detach().numpy())
            batch_labels.append(pred["labels"].detach().numpy())
            batch_scores.append(pred["scores"].detach().numpy())

        # Create output tensors. You need pb_utils.Tensor
        # objects to create pb_utils.InferenceResponse.
        # FIXME: Optimize in the future to avoid unnecessary copy
        batch_boxes_np = np.stack(batch_boxes)
        batch_labels_np = np.stack(batch_labels)
        batch_scores_np = np.stack(batch_scores)
        outputs = {
            "BOXES": batch_boxes_np,
            "LABELS": batch_labels_np,
            "SCORES": batch_scores_np,
        }

        return {"outputs": outputs}

    def metadata(self) -> Dict[str, Any]:
        if hasattr(self, "args"):
            args = self.args
        else:
            args = {}

        return {
            "inputs": [
                {"name": "BATCH_IMAGE", "datatype": "FP32", "shape": [3, -1, -1]},
            ],
            "outputs": [
                {"name": "BOXES", "datatype": "FP32", "shape": [-1, 4]},
                {"name": "LABELS", "datatype": "INT32", "shape": [-1]},
                {"name": "SCORES", "datatype": "FP32", "shape": [-1]},
            ],
            "args": args,
            "max_batch_size": 1,
        }


# class NMS(BuiltinModule):
#     """
#     Non-maximum suppression
#     """

#     async def initialize(self, args: Dict[str, Any]) -> None:
#         self.threshold = args["threshold"]

#     async def execute(self, requests: Any) -> Any:
#         # If request has threshold, use it, otherwise use self.threshold
#         raise NotImplementedError

#     def metadata(self) -> Dict[str, Any]:
#         raise NotImplementedError


# class Crop(BuiltinModule):
#     async def initialize(self, args: Dict[str, Any]) -> None:
#         pass

#     async def execute(self, requests: Any) -> Any:
#         raise NotImplementedError

#     def metadata(self) -> Dict[str, Any]:
#         raise NotImplementedError
