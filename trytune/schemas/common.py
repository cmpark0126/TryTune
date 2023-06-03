from pydantic import BaseModel
from typing import Any


class InferData(BaseModel):
    inputs: Any
    outputs: Any
