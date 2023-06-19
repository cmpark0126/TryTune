import traceback
from typing import Any, Dict

from fastapi import HTTPException
import numpy as np

from trytune.services.moduels import modules
from trytune.services.schedulers import scheduler

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


async def infer_module(module: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    try:
        metadata = modules.get(module)["metadata"]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Module {module} not found.")

    if "max_batch_size" in metadata and metadata["max_batch_size"] > 0:
        use_dynamic_batching = True
    else:
        use_dynamic_batching = False

    _metadata: Dict[str, Any] = {"inputs": {}, "outputs": {}}
    try:
        for input in metadata["inputs"]:
            _metadata["inputs"][input["name"]] = input
        for output in metadata["outputs"]:
            _metadata["outputs"][output["name"]] = output

        _inputs: Dict[str, np.ndarray] = {}
        for name, data in inputs.items():
            datatype = _metadata["inputs"][name]["datatype"]
            _inputs[name] = data.astype(to_numpy_dtype(datatype))

        validate(
            _inputs,
            _metadata["inputs"],
            use_dynamic_batching,
        )
    except Exception:
        raise HTTPException(
            status_code=400, detail=f"While validating inputs: {traceback.format_exc()}"
        )

    try:
        outputs = await scheduler.infer(module, _inputs)
    except Exception:
        raise HTTPException(status_code=400, detail=f"While infering: {traceback.format_exc()}")

    try:
        validate(outputs, _metadata["outputs"], use_dynamic_batching)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"While validating outputs: {traceback.format_exc()}",
        )

    return outputs
