from fastapi import APIRouter
from typing import Any
from trytune.schemas.scheduler import SetSchedulerSchema

router = APIRouter()


@router.post("/schedulers/{scheduler}/set")
async def set_scheduler(scheduler: str, schema: SetSchedulerSchema) -> Any:
    return schema.dict()
