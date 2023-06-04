import httpx
from fastapi import APIRouter, HTTPException
from typing import Any
from trytune.schemas import common, model
from trytune.services.models import Models
from trytune.routers.scheduler import scheduler


router = APIRouter()
models = Models()


@router.get("/models/{model}/metadata")
async def get_metadata(model: str) -> Any:
    try:
        return models.get_metadata(model)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model {model} not found.")


async def get_metadata_from_url(model: str, url: str) -> Any:
    async with httpx.AsyncClient() as client:
        tgt_url = url + f"/v2/models/{model}"
        response = await client.get(tgt_url)

        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Error: {response.text} from {url} with {tgt_url}",
            )
        metadata = response.json()
        return metadata


@router.post("/models/add")
async def add_model(schema: model.AddModelSchema) -> Any:
    if schema.name in models.models:
        raise HTTPException(status_code=400, detail=f"Model {model} already exists.")

    # Send the request to the triton server to get model metadata
    if len(schema.urls) == 0:
        raise HTTPException(status_code=400, detail="No links provided.")

    # Request to triton server to get model metadata
    urls = [url for _instance_type, url in schema.urls.items()]
    metadata = await get_metadata_from_url(schema.name, urls[0])

    for url in urls[1:]:
        other = await get_metadata_from_url(schema.name, url)
        if metadata != other:
            raise HTTPException(
                status_code=400,
                detail=f"Model metadata mismatch: {urls[0]}'s {metadata}, {url}'s {other}",
            )

    # add model to model registry
    models.add(schema.name, {"urls": schema.urls, "metadata": metadata})

    # Return the response with the stored information
    return metadata


@router.post("/models/{model}/infer")
async def infer(model: str, schema: common.InferSchema) -> Any:
    if model != schema.target:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model} does not match the target {schema.target}",
        )

    try:
        _metadata = models.get_metadata(model)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model {model} not found.")

    outs = await scheduler.infer(schema)

    # TODO: send the request to scheduler services and return results
    try:
        return outs
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
