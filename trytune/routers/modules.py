import traceback
from typing import Any, Dict, Union

from fastapi import APIRouter, HTTPException
import httpx
import numpy as np
import tritonclient.http.aio as httpclient

from trytune.schemas import common, module
from trytune.services.moduels import modules
from trytune.services.schedulers import scheduler
from trytune.services.common import OutputTensors

router = APIRouter()

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


def check_datatypes(data: dict) -> None:
    inputs = data.get("inputs", [])
    outputs = data.get("outputs", [])

    for input_data in inputs:
        datatype = input_data.get("datatype")
        if datatype not in DATATYPES:
            raise Exception(f"Unsupported datatype {datatype}")

    for output_data in outputs:
        datatype = output_data.get("datatype")
        if datatype not in DATATYPES:
            raise Exception(f"Unsupported datatype {datatype}")


@router.get("/modules/avaliable_builtins")
async def get_available_builtins() -> Any:
    return modules.available_builtins


@router.get("/modules/list")
async def get_list() -> Any:
    data = {}
    for name, module in modules.modules.items():
        data[name] = module["metadata"]
    return data


# FIXME: we plan to use tritonclient.http.aio in the future
async def get_metadata_from_url(module: str, url: str) -> Any:
    async with httpx.AsyncClient() as client:
        tgt_url = url + f"/v2/models/{module}"
        response = await client.get(tgt_url)

        if response.status_code != 200:
            raise Exception(f"Error: {response.text} from {url} with {tgt_url}")
        metadata = response.json()
        return metadata


async def add_triton_module(schema: module.AddModuleSchema) -> Any:
    # Send the request to the triton server to get module metadata
    if schema.urls is None or len(schema.urls) == 0:
        raise HTTPException(status_code=400, detail="No links provided.")

    # Request to triton server to get module metadata
    urls = [url for _instance_type, url in schema.urls.items()]
    try:
        metadata = await get_metadata_from_url(schema.name, urls[0])
        check_datatypes(metadata)
    except Exception:
        raise HTTPException(
            status_code=400, detail=f"While getting metadata: {traceback.format_exc()}"
        )

    for url in urls[1:]:
        try:
            other = await get_metadata_from_url(schema.name, url)
        except Exception:
            raise HTTPException(
                status_code=400, detail=f"While getting metadata: {traceback.format_exc()}"
            )

        if metadata != other:
            raise HTTPException(
                status_code=400,
                detail=f"Module metadata mismatch: {urls[0]}'s {metadata}, {url}'s {other}",
            )

    # add module to module registry
    metadata["urls"] = schema.urls
    metadata["type"] = schema.type
    modules.set(schema.name, {"metadata": metadata})

    # Return the response with the stored information
    return metadata


async def add_builtin_module(schema: module.AddModuleSchema) -> Any:
    if schema.builtin_args is None or len(schema.builtin_args) == 0:
        raise HTTPException(status_code=400, detail="No builtin_args provided.")

    instance = modules.available_builtins[schema.builtin_args["target"]]["object"]()
    await instance.initialize(schema.builtin_args)
    metadata = instance.metadata()
    metadata["type"] = schema.type

    modules.set(schema.name, {"instance": instance, "metadata": metadata})
    return metadata


@router.post("/modules/add")
async def add_module(schema: module.AddModuleSchema) -> Any:
    if schema.name in modules.modules:
        raise HTTPException(status_code=400, detail=f"Module {module} already exists.")

    if schema.type == module.ModuleTypeSchema.TRITON:
        return await add_triton_module(schema)
    elif schema.type == module.ModuleTypeSchema.BUILTIN:
        return await add_builtin_module(schema)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported module type {schema.type}")


# TODO: dynamic shape validation also needs to be done
def validate(
    tensors: Union[Dict[str, np.ndarray], Dict[str, OutputTensors]],
    metadata: Dict[str, Any],
    use_dynamic_batching: bool,
) -> None:
    for name, item in tensors.items():
        if isinstance(item, OutputTensors):
            _tensors = item.tensors
        else:
            _tensors = [item]

        datatype = to_numpy_dtype(metadata[name]["datatype"])
        for i, tensor in enumerate(_tensors):
            if tensor.dtype != datatype:
                raise Exception(
                    f"Tensor {name}:{i} datatype mismatch: {tensor.dtype} vs {datatype}"
                )

            if use_dynamic_batching:
                tensor_shape = tensor.shape[1:]
            else:
                tensor_shape = tensor.shape
            shape = metadata[name]["shape"]
            if len(tensor_shape) != len(shape):
                raise Exception(
                    f"Tensor {name}:{i} shape mismatch: {tensor_shape} vs {shape} on use_dynamic_batching {use_dynamic_batching}"
                )

            for j in zip(tensor_shape, shape):
                if j[1] == -1:
                    continue
                if j[0] != j[1]:
                    raise Exception(
                        f"Tensor {name}:{i} shape mismatch: {tensor_shape} vs {shape} on use_dynamic_batching {use_dynamic_batching}"
                    )
    pass


@router.get("/modules/{module}/metadata")
async def get_metadata(module: str) -> Any:
    try:
        return modules.get(module)["metadata"]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Module {module} not found.")


@router.post("/modules/{module}/infer")
async def infer(module: str, schema: common.InferSchema) -> Any:
    if module != schema.target:
        raise HTTPException(
            status_code=400,
            detail=f"Module {module} does not match the target inside the schema {schema.target}",
        )

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

        inputs: Dict[str, np.ndarray] = {}
        for name, input in schema.inputs.items():
            datatype = _metadata["inputs"][name]["datatype"]
            if input.shape is not None:
                shape = input.shape
            else:
                shape = _metadata["inputs"][name]["shape"]

            inputs[name] = np.array(input.data, dtype=to_numpy_dtype(datatype)).reshape(shape)

        validate(
            inputs,
            _metadata["inputs"],
            use_dynamic_batching,
        )
    except Exception:
        raise HTTPException(
            status_code=400, detail=f"While validating inputs: {traceback.format_exc()}"
        )

    try:
        outputs = await scheduler.infer(module, inputs)
    except Exception:
        raise HTTPException(status_code=400, detail=f"While infering: {traceback.format_exc()}")

    try:
        validate(outputs, _metadata["outputs"], use_dynamic_batching)
    except Exception:
        raise HTTPException(
            status_code=400, detail=f"While validating outputs: {traceback.format_exc()}"
        )

    response = {}
    for name, output in outputs.items():
        response[name] = output.tolist()

    return response
