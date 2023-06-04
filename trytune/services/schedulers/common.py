import httpx
from abc import ABC, abstractmethod
from typing import Any, List, Dict
import tritonclient.http.aio as httpclient
from trytune.schemas.common import InferSchema, DataSchema


class SchedulerInner(ABC):
    @abstractmethod
    async def infer(self, schema: InferSchema) -> List[DataSchema]:
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


async def infer_with_triton(
    triton_client: httpclient.InferenceServerClient,
    model_metadata: Dict[str, Any],
    inputs: List[DataSchema],
) -> Any:
    """
    Request to triton server to infer the model with the given inputs.

    References:
        https://github.com/triton-inference-server/client/blob/main/src/python/examples/simple_http_aio_infer_client.py
    """
    infer_inputs: List[httpclient.InferInput] = []
    infer_requested_outputs: List[httpclient.InferRequestedOutput] = []

    for input_metadata in model_metadata["inputs"]:
        name = input_metadata["name"]
        shape = input_metadata["shape"]
        datatype = input_metadata["datatype"]

        infer_input = httpclient.InferInput(name, shape, datatype)

        # FIXME: numpy array is not supported yet
        infer_input.set_data_from_numpy(inputs[name].data, binary_data=False)
        infer_inputs.append(infer_input)

    for output_metadata in model_metadata["outputs"]:
        name = output_metadata["name"]
        infer_requested_output = httpclient.InferRequestedOutput(name, binary_data=True)
        infer_requested_outputs.append(infer_requested_output)

    return await triton_client.infer(
        model_metadata["name"],
        infer_inputs,
        outputs=infer_requested_outputs,
    )
