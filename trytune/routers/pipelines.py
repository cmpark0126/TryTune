from fastapi import APIRouter
from trytune.schemas import common


router = APIRouter()


@router.get("/pipelines/{pipeline}")
async def get_metadata(pipeline: str):
    dummy = {
        "name": pipeline,
        "inputs": [],
        "outputs": [],
        "tensors": [],
        "models": [],
    }
    return dummy


@router.post("/pipelines/{pipeline}/infer")
async def infer(pipeline: str, infer_data: common.InferData):
    print(f"Received request for model {pipeline} with data: {infer_data}")
    return infer_data
