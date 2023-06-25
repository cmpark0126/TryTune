from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, validator


class ModuleTypeSchema(str, Enum):
    TRITON = "triton"
    BUILTIN = "builtin"
    # PYTORCH = "pytorch"
    # TENSORFLOW = "tensorflow"
    # ONNX = "onnx"
    # MXNET = "mxnet"
    # CNTK = "cntk"
    # CAFFE2 = "caffe2"
    # CUSTOM = "custom"


class AddModuleSchema(BaseModel):
    """
    Schema for adding module.

    Attributes:
        urls (dict): A dictionary of triton server urls for each instance type.

    Example:
        If user sends a request to add a module with the following information:
        {
            "name": "resnet50",
            "urls": {
                "g4dn.xlarge": "http://eks.ingress.url/g4dn"
                "g5.xlarge": "http://eks.ingress.url/g5"
                "inf1.xlarge": "http://eks.ingress.url/inf1"
                ...
            }
        }
        FYI, all urls are linked to the triton servers serving same module but run on different instance types

        Then the module registry will store the following information
        obtained from the triton server:
        {
            "name": "resnet50",
            "inputs": [
                {"name": "input__0", "datatype": "FP32", "shape": [3, 224, 224]}
            ],
            "outputs": [
                {"name": "output__0", "datatype": "FP32", "shape": [1000]}
            ],
        }

        Finnaly, send the above information back to the user also.

    Warning:
        URL should start with "http://" or "https://"

    """

    name: str
    type: ModuleTypeSchema
    urls: Optional[Dict[str, str]] = None
    builtin_args: Optional[Dict[str, Any]] = None

    @validator("urls")
    def validate_urls(cls, urls, values):  # type: ignore
        type = values.get("type")
        if type == ModuleTypeSchema.TRITON:
            if urls is None:
                raise ValueError("urls should not be None for triton module")

            if len(urls) == 0:
                raise ValueError("urls should not be empty for triton module")

            cond = all(
                url.startswith("http://") or url.startswith("https://")
                for url in urls.values()
            )
            if not cond:
                raise ValueError("all urls should start with http:// or https://")

        return urls

    @validator("builtin_args")
    def validate_builtin_args(cls, builtin_args, values):  # type: ignore
        type = values.get("type")
        if type == ModuleTypeSchema.BUILTIN:
            if builtin_args is None:
                raise ValueError("builtin_args should not be None for builtin module")

            if len(builtin_args) == 0:
                raise ValueError("builtin_args should not be empty for builtin module")

        return builtin_args
