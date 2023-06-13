from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
import tritonclient.http.aio as httpclient

from trytune.schemas.module import ModuleTypeSchema
from trytune.services.common import OutputTensors
from trytune.services.moduels import modules


class SchedulerInner(ABC):
    @abstractmethod
    async def infer(self, module: str, inputs: Dict[str, np.ndarray]) -> Dict[str, OutputTensors]:
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
    module_name: str,
    module: Dict[str, Any],
    inputs: Dict[str, np.ndarray],
    url: str,
) -> Dict[str, OutputTensors]:
    """
    Request to triton server to infer the module with the given inputs.

    References:
        https://github.com/triton-inference-server/client/blob/main/src/python/examples/simple_http_aio_infer_client.py
    """
    infer_inputs: List[httpclient.InferInput] = []
    infer_requested_outputs: List[httpclient.InferRequestedOutput] = []

    module_metadata = module["metadata"]

    for input_metadata in module_metadata["inputs"]:
        name = input_metadata["name"]
        if name not in inputs:
            raise ValueError(f"input {name} is not provided for module {module_name}")

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

    outputs: Dict[str, OutputTensors] = {}
    for output_metadata in module_metadata["outputs"]:
        name = output_metadata["name"]
        outputs[name] = OutputTensors(result.as_numpy(name))

    return outputs


async def infer_with_builtin(
    module_name: str,
    module: Dict[str, Any],
    inputs: Dict[str, np.ndarray],
) -> Dict[str, OutputTensors]:
    module_metadata = module["metadata"]
    for input_metadata in module_metadata["inputs"]:
        name = input_metadata["name"]
        if name not in inputs:
            raise ValueError(f"input {name} is not provided for module {module_name}")

    instance = module["instance"]
    request = {"inputs": inputs}
    response = await instance.execute(request)
    return response["outputs"]


async def infer(
    module_name: str, inputs: Dict[str, np.ndarray], **kwargs: Any
) -> Dict[str, OutputTensors]:
    module = modules.get(module_name)
    metadata = module["metadata"]
    module_type: ModuleTypeSchema = metadata["type"]
    if module_type == ModuleTypeSchema.TRITON:
        if "instance_type" not in kwargs:
            raise ValueError("instance_type should not be None for triton module")

        url = metadata["urls"][kwargs["instance_type"]]
        return await infer_with_triton(module_name, module, inputs, url)
    elif module_type == ModuleTypeSchema.BUILTIN:
        return await infer_with_builtin(module_name, module, inputs)
    else:
        raise ValueError(f"module type {module_type} is not supported")
