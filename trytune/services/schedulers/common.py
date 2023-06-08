from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
import tritonclient.http.aio as httpclient

from trytune.schemas.common import DataSchema, InferSchema
from trytune.schemas.module import ModuleTypeSchema


class SchedulerInner(ABC):
    @abstractmethod
    async def infer(self, module: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        raise NotImplementedError("infer is not implemented")

    @abstractmethod
    async def start(self) -> None:
        raise NotImplementedError("infer is not implemented")

    @abstractmethod
    async def stop(self) -> None:
        raise NotImplementedError("infer is not implemented")

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError("infer is not implemented")


def get_numpy_dtype(datatype: str) -> Any:
    if datatype == "FP32":
        return np.float32
    else:
        raise NotImplementedError(f"datatype {datatype} is not supported")


async def infer_with_triton(
    url: str,
    module_metadata: Dict[str, Any],
    inputs: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Request to triton server to infer the module with the given inputs.

    References:
        https://github.com/triton-inference-server/client/blob/main/src/python/examples/simple_http_aio_infer_client.py
    """
    infer_inputs: List[httpclient.InferInput] = []
    infer_requested_outputs: List[httpclient.InferRequestedOutput] = []

    for input_metadata in module_metadata["inputs"]:
        name = input_metadata["name"]
        shape = input_metadata["shape"]
        datatype = input_metadata["datatype"]

        infer_input = httpclient.InferInput(name, shape, datatype)

        # FIXME: numpy array is not supported yet
        # FIXME: various datatype in the future
        data = np.array(inputs[name].data, dtype=get_numpy_dtype(datatype)).reshape(shape)
        infer_input.set_data_from_numpy(data, binary_data=True)
        infer_inputs.append(infer_input)

    for output_metadata in module_metadata["outputs"]:
        name = output_metadata["name"]
        infer_requested_output = httpclient.InferRequestedOutput(name, binary_data=True)
        infer_requested_outputs.append(infer_requested_output)

    # FIXME: use ssl to get security
    parsed_url = urlparse(url)
    triton_client = httpclient.InferenceServerClient(url=parsed_url.netloc + parsed_url.path)
    result = await triton_client.infer(
        module_metadata["name"],
        inputs=infer_inputs,
        outputs=infer_requested_outputs,
    )

    outputs: Dict[str, np.ndarray] = {}
    for output_metadata in module_metadata["outputs"]:
        name = output_metadata["name"]
        outputs[name] = result.as_numpy(name)

    return outputs


async def infer_with_builtin(
    module_metadata: Dict[str, Any],
    inputs: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    raise NotImplementedError("infer_with_builtin is not implemented")


async def infer(
    module_metadata: Dict[str, Any],
    inputs: Dict[str, np.ndarray],
    instance_type: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    module_type: ModuleTypeSchema = module_metadata["type"]
    if module_type == ModuleTypeSchema.TRITON:
        if instance_type is None:
            raise ValueError("instance_type should not be None for triton module")

        url = module_metadata["urls"][instance_type]
        return await infer_with_triton(url, module_metadata, inputs)
    elif module_type == ModuleTypeSchema.BUILTIN:
        if instance_type is not None:
            # TODO: change to logger
            print("instance_type is ignored for builtin module")
        return await infer_with_builtin(module_metadata, inputs)
    else:
        raise ValueError(f"module type {module_type} is not supported")
