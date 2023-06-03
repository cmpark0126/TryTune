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
async def infer(pipeline: str, infer_data: common.InferData):
    print(f"Received request for model {pipeline} with data: {infer_data}")
    return infer_data
