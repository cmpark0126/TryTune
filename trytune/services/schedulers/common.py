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


# # TODO: generalize input format
# async def infer_with_triton(clents: httpclient.InferenceServerClient, inputs: []) -> Any:
#     inputs = []
#     outputs = []
#     inputs.append(httpclient.InferInput("INPUT0", [1, 16], "INT32"))
#     inputs.append(httpclient.InferInput("INPUT1", [1, 16], "INT32"))

#     # Initialize the data
#     inputs[0].set_data_from_numpy(input0_data, binary_data=False)
#     inputs[1].set_data_from_numpy(input1_data, binary_data=True)

#     outputs.append(httpclient.InferRequestedOutput("OUTPUT0", binary_data=True))
#     outputs.append(httpclient.InferRequestedOutput("OUTPUT1", binary_data=False))
#     query_params = {"test_1": 1, "test_2": 2}
#     results = await triton_client.infer(
#         model_name,
#         inputs,
#         outputs=outputs,
#         query_params=query_params,
#         headers=headers,
#         request_compression_algorithm=request_compression_algorithm,
#         response_compression_algorithm=response_compression_algorithm,
#     )
