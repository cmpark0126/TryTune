from typing import Any, Dict

import numpy as np
import torch

from trytune.services.moduels.common import BuiltinModule


# NOTE: https://github.com/triton-inference-server/tutorials/blob/main/Quick_Deploy/PyTorch/export.py
class Resnet50FromTorchHub(BuiltinModule):
    async def initialize(self, args: Dict[str, Any]) -> None:
        self.args = args

        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet50", pretrained=True
        ).eval()

    async def execute(self, requests: Any) -> Any:
        inputs = requests["inputs"]
        input__0 = torch.from_numpy(inputs["input__0"])

        output__0 = self.model(input__0)

        output__0 = output__0.detach().numpy().astype(np.float32).reshape((-1, 1000))
        outputs = {"output__0": output__0}

        return {"outputs": outputs}

    def metadata(self) -> Dict[str, Any]:
        if hasattr(self, "args"):
            args = self.args
        else:
            args = {"upscale_factor": "int"}

        return {
            "inputs": [
                {"name": "input__0", "datatype": "FP32", "shape": [1, 3, -1, -1]},
            ],
            "outputs": [
                {"name": "output__0", "datatype": "FP32", "shape": [1, 1000]},
            ],
            "args": args,
            "max_batch_size": 0,
        }
