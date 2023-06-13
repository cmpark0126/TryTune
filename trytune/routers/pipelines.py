from typing import Any

from fastapi import APIRouter, HTTPException

from trytune.schemas import common

router = APIRouter()


@router.delete("/pipelines/clear")
async def clear() -> Any:
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/pipelines/{pipeline}")
async def get_metadata(pipeline: str) -> Any:
    dummy = {
        "name": pipeline,
        "inputs": [],
        "outputs": [],
        "tensors": [],
        "modules": [],
    }
    return dummy


# TODO: during add, we need to check all tensors and modules are valid


@router.post("/pipelines/infer")
async def infer(infer: common.InferSchema) -> Any:
    print(f"Received request for pipeline {infer.target} with data: {infer.inputs}")
    return infer
