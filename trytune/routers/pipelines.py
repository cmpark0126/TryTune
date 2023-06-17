from typing import Any

from fastapi import APIRouter, HTTPException

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
            input_tensors.add(stage.inputs[input["name"]])
        for output in module_metadata["outputs"]:
            if output["name"] not in stage.outputs:
                raise HTTPException(
                    status_code=400,
                    detail=f"Module {stage.module} output {output['name']} not found.",
                )
            output_tensors.add(stage.outputs[output["name"]])

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
    raise HTTPException(status_code=501, detail="Not implemented yet.")
