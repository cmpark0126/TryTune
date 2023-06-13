from typing import Any, Dict, Optional

import numpy as np
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

from trytune.services.common import OutputTensor
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
            batch_boxes.append(pred["boxes"].detach().numpy().astype(np.float32))
            batch_labels.append(pred["labels"].detach().numpy().astype(np.int32))
            batch_scores.append(pred["scores"].detach().numpy().astype(np.float32))

        # Create output tensors. You need pb_utils.Tensor
        # objects to create pb_utils.InferenceResponse.
        # FIXME: Optimize in the future to avoid unnecessary copy
        batch_boxes_np = np.stack(batch_boxes)
        batch_labels_np = np.stack(batch_labels)
        batch_scores_np = np.stack(batch_scores)
        outputs = {
            "BOXES": OutputTensor(batch_boxes_np),
            "LABELS": OutputTensor(batch_labels_np),
            "SCORES": OutputTensor(batch_scores_np),
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


class Crop(BuiltinModule):
    async def initialize(self, args: Dict[str, Any]) -> None:
        if "label" in args:
            self.label: Optional[int] = int(args["label"])
        else:
            args["label"] = None
            self.label = None

        if "threshold" in args:
            self.threshold = float(args["threshold"])
        else:
            args["threshold"] = 0.9
            self.threshold = 0.9

        if "max_nums" in args:
            self.max_nums: Optional[int] = int(args["max_nums"])
        else:
            args["max_nums"] = None
            self.max_nums = None

        self.args = args
        pass

    async def execute(self, requests: Any) -> Any:
        inputs = requests["inputs"]
        image = inputs["IMAGE"]
        boxes = inputs["BOXES"]
        labels = inputs["LABELS"]
        scores = inputs["SCORES"]

        indices = scores >= self.threshold
        pred_boxes = boxes[indices]
        pred_labels = labels[indices]

        if self.label is not None:
            pred_boxes = [
                box for box, label in zip(pred_boxes, pred_labels) if label == self.label
            ]

        # FIXME: remove transform after supporing dynamic pipelines
        outputs = []
        for i, box in enumerate(pred_boxes):
            if self.max_nums is not None and i >= self.max_nums:
                break
            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2])
            y_max = int(box[3])
            cropped = image[:, y_min:y_max, x_min:x_max]
            outputs.append(cropped)

        return {"outputs": {"CROPPED_IMAGES": OutputTensor(outputs)}}

    def metadata(self) -> Dict[str, Any]:
        if hasattr(self, "args"):
            args = self.args
        else:
            args = {
                "label": "Optional[int]",
                "threshold": "Optional[int]",
                "max_nums": "Optional[int]",
            }

        return {
            "inputs": [
                {"name": "IMAGE", "datatype": "FP32", "shape": [3, -1, -1]},
                {"name": "BOXES", "datatype": "FP32", "shape": [-1, 4]},
                {"name": "LABELS", "datatype": "FP32", "shape": [-1]},
                {"name": "SCORES", "datatype": "FP32", "shape": [-1]},
            ],
            "outputs": [
                {"name": "CROPPED_IMAGES", "datatype": "FP32", "shape": [3, -1, -1]},
            ],
            "args": args,
            "max_batch_size": 0,
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
