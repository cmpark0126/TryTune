from typing import Any, Dict, List, Optional

from pydantic import BaseModel, validator


class DataSchema(BaseModel):
    data: List[Any]
    shape: Optional[List[int]] = None


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
                "i1": {"data": [1.0, 2.0, 3.0], "shape": [3, 224, 224]},
                "i2": {"data": [1.0, 2.0, 3.0], "shape": [3, 224, 224]},
            ]
        }
    """

    target: str
    inputs: Dict[str, DataSchema]
