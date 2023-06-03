from pydantic import BaseModel
from typing import List, Dict


class TensorSchema(BaseModel):
    name: str


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
    Schema for adding a pipeline.

    Attributes:
        name (str): The name of the pipeline.
        tensors (dict): Dictionary containing input and output tensors information.
        models (list): List of model information.

    Example:
        {
            "name": "pipe1",
            "tensors": {
                "inputs": [
                    {"name": "pinput__0"}
                ],
                "outputs": [
                    {"name": "poutput__0"}
                ],
                "tensors": []
            },
            "stages": [
                {
                    "name": "target",
                    "model": "resnet50",
                    "inputs": [{"src": "input__0", "tgt": "pinput__0"}],
                    "outputs": [{"src": "output__0", "tgt": "poutput__0"}]
                }
            ]
        }
    """

    name: str
    tensors: Dict[str, List[TensorSchema]]
    stages: List[StageSchema]
