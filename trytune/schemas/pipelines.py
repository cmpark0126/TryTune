from pydantic import BaseModel
from typing import List


class TensorSchema(BaseModel):
    name: str


class TensorsSchema(BaseModel):
    inputs: List[TensorSchema]
    outputs: List[TensorSchema]
    interms: List[TensorSchema]


class StageIO(BaseModel):
    src: str
    tgt: str


class StageSchema(BaseModel):
    name: str
    model: str
    inputs: List[StageIO]
    outputs: List[StageIO]


class PipelineAddSchema(BaseModel):
    """
    Schema for adding a pipeline. Can be formed by DAG.

    Attributes:
        name (str): The name of the pipeline.
        tensors (dict): Dictionary containing input and output tensors information.
        models (list): List of model information.

    Example:
        {
            "name": "pipe1",
            "tensors": {
                "inputs": [{"name": "pinput__0"}],
                "outputs": [{"name": "poutput__0"}],
                "interms": [{"name": "pinterm__0"}],
            },
            # pinput__0    -> [classifier] -> pinterm__0          -> [selector] -> poutput__0
            # input_tensor -> [stage]      -> intermediate_tensor -> [stage]    -> output_tensor
            "stages": [
                {
                    "name": "classifier",
                    "model": "resnet50",
                    "inputs": [{"src": "input__0", "tgt": "pinput__0"}],
                    "outputs": [{"src": "output__0", "tgt": "pinterm__0"}],
                },
                {
                    "name": "selector",
                    "model": "top_five",
                    "inputs": [{"src": "input__0", "tgt": "pinterm__0"}],
                    "outputs": [{"src": "output__0", "tgt": "poutput__0"}],
                },
            ],
        }
    """

    name: str
    tensors: TensorsSchema
    stages: List[StageSchema]
