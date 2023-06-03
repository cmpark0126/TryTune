from fastapi import APIRouter
from typing import Any
from trytune.schemas import common

router = APIRouter()


@router.get("/pipelines/{pipeline}")
async def get_metadata(pipeline: str) -> Any:
    dummy = {
        "name": pipeline,
        "inputs": [],
        "outputs": [],
        "tensors": [],
        "models": [],
    }
    return dummy


# TODO: during add, we need to check all tensors and modules are valid


@router.post("/pipelines/{pipeline}/infer")
async def infer(pipeline: str, infer: common.InferSchema) -> Any:
    print(f"Received request for model {pipeline} with data: {infer}")
    return infer
