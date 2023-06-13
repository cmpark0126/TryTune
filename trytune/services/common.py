from typing import Any, List, Union

import numpy as np


class OutputTensor:
    def __init__(self, data: Union[np.ndarray, List[np.ndarray]]):
        if isinstance(data, list):
            self.tensors = data
        else:
            self.tensors = [data]

    def __repr__(self):  # type: ignore
        return f"OutputTensor(tensors={self.tensors})"

    def tolist(self) -> List[List[Any]]:
        return [x.tolist() for x in self.tensors]
