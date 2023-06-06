from pydantic import BaseModel
from typing import List, Any, Dict


class DataSchema(BaseModel):
    data: List[Any]


class InferSchema(BaseModel):
    """
    Data to be used for target program(pipeline or module) inference.

    Attributes:
        target (str): The name of the target program.
        inputs (list): The input datas to be used for inference.

    Example:
        {
            "target": "pipe1",
            "inputs": [
                "i1": {"data": [1.0, 2.0, 3.0]},
                "i2": {"data": [1.0, 2.0, 3.0]},
            ]
        }
    """

    target: str
    # TODO: add target type enum pipeline/module
    inputs: Dict[str, DataSchema]
