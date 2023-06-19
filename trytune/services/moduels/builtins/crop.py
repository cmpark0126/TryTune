from typing import Any, Dict, Optional

import cv2
import numpy as np

from trytune.services.moduels.common import BuiltinModule


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

        if "mode" in args:
            self.mode: str = str(args["mode"])
        else:
            args["mode"] = "pad"
            self.mode = "pad"

        self.args = args
        pass

    async def execute(self, requests: Any) -> Any:
        inputs = requests["inputs"]
        image = inputs["IMAGE"][0]
        boxes = inputs["BOXES"][0]
        labels = inputs["LABELS"][0]
        scores = inputs["SCORES"][0]

        indices = scores >= self.threshold
        pred_boxes = boxes[indices]
        pred_labels = labels[indices]

        if self.label is not None:
            pred_boxes = [
                box
                for box, label in zip(pred_boxes, pred_labels)
                if label == self.label
            ]

        # FIXME: remove transform after supporing dynamic pipelines
        outputs = []
        whs = []
        max_w = 0
        max_h = 0
        for i, box in enumerate(pred_boxes):
            if self.max_nums is not None and i >= self.max_nums:
                break
            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2])
            y_max = int(box[3])
            cropped = image[:, y_min:y_max, x_min:x_max]
            outputs.append(cropped)

            w = x_max - x_min
            if max_w < w:
                max_w = w
            h = y_max - y_min
            if max_h < h:
                max_h = h
            whs.append(np.array([w, h]))

        if self.mode == "pad":
            _outputs = [
                np.pad(
                    output,
                    (
                        (0, 0),
                        (0, max_h - output.shape[1]),
                        (0, max_w - output.shape[2]),
                    ),
                    mode="constant",
                )
                for output in outputs
            ]
        elif self.mode == "resize":
            _outputs = [
                np.transpose(
                    cv2.resize(np.transpose(output, (1, 2, 0)), (max_w, max_h)),
                    (2, 0, 1),
                )
                for output in outputs
            ]
        else:
            raise ValueError(f"Invalid mode {self.mode}")

        return {
            "outputs": {
                "CROPPED_IMAGES": np.stack(_outputs).astype(np.float32),
                "WHS": np.stack(whs).astype(np.int32),
            }
        }

    def metadata(self) -> Dict[str, Any]:
        if hasattr(self, "args"):
            args = self.args
        else:
            args = {
                "label": "Optional[int]",
                "threshold": "Optional[int]",
                "max_nums": "Optional[int]",
                "mode": "pad or resize",
            }

        return {
            "inputs": [
                {"name": "IMAGE", "datatype": "FP32", "shape": [1, 3, -1, -1]},
                {"name": "BOXES", "datatype": "FP32", "shape": [1, -1, 4]},
                {"name": "LABELS", "datatype": "FP32", "shape": [1, -1]},
                {"name": "SCORES", "datatype": "FP32", "shape": [1, -1]},
            ],
            "outputs": [
                {
                    "name": "CROPPED_IMAGES",
                    "datatype": "FP32",
                    "shape": [-1, 3, -1, -1],
                },
                {"name": "WHS", "datatype": "INT32", "shape": [-1, 2]},
            ],
            "args": args,
            "max_batch_size": 0,
        }
