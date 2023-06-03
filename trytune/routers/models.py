from fastapi import APIRouter
from trytune.schemas import common

router = APIRouter()


@router.get("/models/{model}")
async def get_metadata(model: str):
    # TODO: obtained from model registry in the future
    dummy = {
        "name": model,
        "inputs": [],
        "outputs": [],
    }
    return dummy


@router.post("/models/{pipeline}/infer")
async def infer(pipeline: str, infer: common.InferSchema):
    print(f"Received request for model {pipeline} with data: {infer}")
    return infer
