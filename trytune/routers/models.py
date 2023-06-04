import httpx
from urllib.parse import urlparse
from fastapi import APIRouter, HTTPException
from typing import Any, List, Dict
import tritonclient.http.aio as httpclient
from trytune.schemas import common, model
from trytune.services.models import Models
from trytune.routers.scheduler import scheduler


router = APIRouter()
models = Models()

DATATYPES = [
    "FP32"
]  # "FP16", "FP32", "FP64", "INT8", "INT16", "INT32", "INT64", "BOOL" are not supported yet


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


@router.get("/models/{model}/metadata")
async def get_metadata(model: str) -> Any:
    try:
        return models.get(model)["metadata"]
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
        check_datatypes(metadata)
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
    for instance_type, url in schema.urls.items():
        url_wo_scheme = urlparse(url).netloc
        # FIXME: use ssl to get security
        triton_client = httpclient.InferenceServerClient(url=url_wo_scheme)
        clents[instance_type] = triton_client
    assert len(clents) == len(schema.urls)
    metadata["urls"] = schema.urls
    models.set(schema.name, {"clents": clents, "metadata": metadata})

    # Return the response with the stored information
    return metadata


def validate(outs: List[common.DataSchema]) -> None:
    # for out in outs:
    #     if out.name != schema.target:
    #         raise Exception(f"Output {out.name} does not match the target {schema.target}")

    #     if out.shape != schema.shape:
    #         raise Exception(f"Output {out.shape} does not match the target {schema.shape}")

    #     if out.datatype != schema.datatype:
    #         raise Exception(f"Output {out.datatype} does not match the target {schema.datatype}")
    pass


@router.post("/models/infer")
async def infer(schema: common.InferSchema) -> Any:
    try:
        _ = models.get(schema.target)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model {model} not found.")

    try:
        validate(schema.inputs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        outs = await scheduler.infer(schema)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        validate(outs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return outs
