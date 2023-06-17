from typing import List

from pydantic import BaseModel


class TensorSchema(BaseModel):
    name: str


class TensorsSchema(BaseModel):
    inputs: List[TensorSchema]
    outputs: List[TensorSchema]


class StageIO(BaseModel):
    src: str
    tgt: str


class StageSchema(BaseModel):
    name: str
    module: str
    inputs: List[StageIO]
    outputs: List[StageIO]


class AddPipelineSchema(BaseModel):
    """
    Schema for adding a pipeline. Can be formed by DAG.

    Attributes:
        name (str): The name of the pipeline.
        tensors (dict): Dictionary containing input and output tensors information.
        modules (list): List of module information.

    Example:
        {
            "name": "pipe1",
            "tensors": {
                "inputs": [{"name": "pinput__0"}],
                "outputs": [{"name": "poutput__0"}],
            },
            # pinput__0    -> [classifier] -> pinterm__0          -> [selector] -> poutput__0
            # input_tensor -> [stage]      -> intermediate_tensor -> [stage]    -> output_tensor
            "stages": [
                {
                    "name": "classifier",
                    "module": "resnet50",
                    "inputs": [{"src": "input__0", "tgt": "pinput__0"}],
                    "outputs": [{"src": "output__0", "tgt": "pinterm__0"}],
                },
                {
                    "name": "selector",
                    "module": "top_five",
                    "inputs": [{"src": "input__0", "tgt": "pinterm__0"}],
                    "outputs": [{"src": "output__0", "tgt": "poutput__0"}],
                },
            ],
        }
    """

    name: str
    tensors: TensorsSchema
    stages: List[StageSchema]
