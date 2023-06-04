import httpx
from fastapi import APIRouter, HTTPException
from typing import Any, List, Dict
import tritonclient.http.aio as httpclient
from trytune.schemas import common, model
from trytune.services.models import Models
from trytune.routers.scheduler import scheduler


router = APIRouter()
models = Models()


@router.get("/models/{model}/metadata")
async def get_metadata(model: str) -> Any:
    try:
        return models.get_metadata(model)["metadata"]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model {model} not found.")


# FIXME: we plan to use tritonclient.http.aio in the future
async def get_metadata_from_url(model: str, url: str) -> Any:
    async with httpx.AsyncClient() as client:
        tgt_url = url + f"/v2/models/{model}"
        response = await client.get(tgt_url)

        if response.status_code != 200:
            raise Exception(f"Error: {response.text} from {url} with {tgt_url}")
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
    try:
        metadata = await get_metadata_from_url(schema.name, urls[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    for url in urls[1:]:
        try:
            other = await get_metadata_from_url(schema.name, url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        if metadata != other:
            raise HTTPException(
                status_code=400,
                detail=f"Model metadata mismatch: {urls[0]}'s {metadata}, {url}'s {other}",
            )

    # add model to model registry
    clents: Dict[str, httpclient.InferenceServerClient] = {}
    # for instance_type, url in schema.urls.items():
    #     triton_client = httpclient.InferenceServerClient(url=url)
    #     clents[instance_type] = triton_client
    # assert len(clents) == len(schema.urls)
    metadata["urls"] = schema.urls
    models.add(schema.name, {"clents": clents, "metadata": metadata})

    # Return the response with the stored information
    return metadata


def validate_outs(outs: List[common.DataSchema]) -> None:
    # for out in outs:
    #     if out.name != schema.target:
    #         raise Exception(f"Output {out.name} does not match the target {schema.target}")

    #     if out.shape != schema.shape:
    #         raise Exception(f"Output {out.shape} does not match the target {schema.shape}")

    #     if out.dtype != schema.dtype:
    #         raise Exception(f"Output {out.dtype} does not match the target {schema.dtype}")
    pass


@router.post("/models/infer")
async def infer(model: str, schema: common.InferSchema) -> Any:
    try:
        _metadata = models.get_metadata(model)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model {model} not found.")

    try:
        outs = await scheduler.infer(schema)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        validate_outs(outs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return outs
