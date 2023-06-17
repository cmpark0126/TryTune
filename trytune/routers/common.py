from typing import Any, Dict

import numpy as np

DATATYPES = [
    "FP32",
    "INT32",
]  # "FP16", "FP32", "FP64", "INT8", "INT16", "INT64", "BOOL" are not supported yet


def to_numpy_dtype(datatype: str) -> Any:
    if datatype == "FP32":
        return np.float32
    elif datatype == "INT32":
        return np.int32
    else:
        raise Exception(f"Unsupported datatype {datatype}")


# TODO: dynamic shape validation also needs to be done
def validate(
    tensors: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    use_dynamic_batching: bool,
) -> None:
    for name, tensor in tensors.items():
        datatype = to_numpy_dtype(metadata[name]["datatype"])
        if tensor.dtype != datatype:
            raise Exception(f"Tensor {name} datatype mismatch: {tensor.dtype} vs {datatype}")

        if use_dynamic_batching:
            tensor_shape = tensor.shape[1:]
        else:
            tensor_shape = tensor.shape
        shape = metadata[name]["shape"]
        if len(tensor_shape) != len(shape):
            raise Exception(
                f"Tensor {name} shape mismatch: {tensor_shape} vs {shape} on use_dynamic_batching {use_dynamic_batching}"
            )

        for i in zip(tensor_shape, shape):
            if i[1] == -1:
                continue
            if i[0] != i[1]:
                raise Exception(
                    f"Tensor {name} shape mismatch: {tensor_shape} vs {shape} on use_dynamic_batching {use_dynamic_batching}"
                )
    pass
