import traceback
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
import numpy as np

from trytune.routers.common import infer_module
from trytune.schemas import common, pipeline
from trytune.services.moduels import modules
from trytune.services.pipelines import pipelines

router = APIRouter()


@router.get("/pipelines/list")
async def get_list() -> Any:
    data = {}
    for name, _pipeline in pipelines.pipelines.items():
        data[name] = _pipeline["metadata"]
    return data


@router.delete("/pipelines/clear")
async def clear() -> Any:
    pipelines.pipelines.clear()
    return {"message": "Pipelines cleared"}


@router.post("/pipelines/add")
async def add_pipeline(schema: pipeline.AddPipelineSchema) -> Any:
    # Check if the pipeline name is already in use
    if schema.name in pipelines.pipelines:
        raise HTTPException(status_code=400, detail=f"Pipeline {schema.name} already exists.")

    # FIXME: check if the pipeline is valid
    input_tensors = set()
    output_tensors = set()
    for stage in schema.stages:
        if stage.module not in modules.modules:
            raise HTTPException(status_code=400, detail=f"Module {stage.module} not found.")

        module_metadata = modules.modules[stage.module]["metadata"]
        for input in module_metadata["inputs"]:
            if input["name"] not in stage.inputs:
                raise HTTPException(
                    status_code=400,
                    detail=f"Module {stage.module} input {input['name']} not found.",
                )
            input_tensors.add(stage.inputs[input["name"]].name)
        for output in module_metadata["outputs"]:
            if output["name"] not in stage.outputs:
                raise HTTPException(
                    status_code=400,
                    detail=f"Module {stage.module} output {output['name']} not found.",
                )
            tensor_name = stage.outputs[output["name"]].name
            if tensor_name in output_tensors:
                raise HTTPException(
                    status_code=400,
                    detail=f"Module {stage.module} output {output['name']} is already used.",
                )
            output_tensors.add(tensor_name)

    for input in schema.tensors.inputs:
        if input.name not in input_tensors:
            raise HTTPException(
                status_code=400,
                detail=f"Input tensor {input.name} not found.",
            )
        if input.name in output_tensors:
            raise HTTPException(
                status_code=400,
                detail=f"Input tensor {input.name} is also an output tensor.",
            )
    for output in schema.tensors.outputs:
        if output.name not in output_tensors:
            raise HTTPException(
                status_code=400,
                detail=f"Output tensor {output.name} not found.",
            )
        # NOTE: intermediate tensors can be reused as pipeline output tensors

    # Add pipeline to pipeline registry
    pipelines.set(schema.name, {"metadata": schema})

    # Return the response with the stored information
    return {"message": f"Pipeline {schema.name} added"}


@router.get("/pipelines/{pipeline}/metadata")
async def get_metadata(pipeline: str) -> Any:
    try:
        return pipelines.get(pipeline)["metadata"]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline} not found.")


@router.post("/pipelines/{pipeline}/infer")
async def infer(pipeline: str, schema: common.InferSchema) -> Any:
    if pipeline != schema.target:
        raise HTTPException(
            status_code=400,
            detail=f"Pipeline {pipeline} does not match the target inside the schema {schema.target}",
        )

    try:
        metadata = pipelines.get(pipeline)["metadata"]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline} not found.")

    _metadata: Dict[str, Any] = {"inputs": {}, "outputs": {}}
    for input in metadata.tensors.inputs:
        _metadata["inputs"][input.name] = input
    for output in metadata.tensors.outputs:
        _metadata["outputs"][output.name] = output

    try:
        tensors: Dict[str, Any] = {}
        for name, input in schema.inputs.items():
            tensors[name] = np.array(input.data)
        for name in _metadata["inputs"].keys():
            if name not in tensors:
                raise HTTPException(
                    status_code=400,
                    detail=f"Input tensor {name} not found.",
                )
    except Exception:
        raise HTTPException(
            status_code=400, detail=f"While validating inputs: {traceback.format_exc()}"
        )

    # TODO: run all modules asynchronously
    # TODO: if dynamic excution occurs, it divides multiple streams
    # TODO: support buisness logic scripts
    for stage in metadata.stages:
        inputs = {}
        for src, dst in stage.inputs.items():
            data = tensors[dst.name]
            if dst.shape is not None:
                data.reshape(dst.shape)
            inputs[src] = data

        # Execute the module
        outputs = await infer_module(stage.module, inputs)

        for src, dst in stage.outputs.items():
            assert dst.name not in tensors
            data = outputs[src]
            if dst.shape is not None:
                data.reshape(dst.shape)
            tensors[dst.name] = outputs[src]

    response = {}
    for name in _metadata["outputs"].keys():
        response[name] = tensors[name].tolist()

    return response
