from fastapi import APIRouter, HTTPException
from typing import Any
from trytune.schemas.scheduler import SetSchedulerSchema
from trytune.services.scheduler import Scheduler

router = APIRouter()
scheduler = Scheduler()


@router.post("/schedulers/{scheduler_name}/set")
async def set_scheduler(scheduler_name: str, schema: SetSchedulerSchema) -> Any:
    if scheduler_name != schema.name:
        raise HTTPException(
            status_code=400,
            detail=f"Scheduler {scheduler_name} does not match the target {schema.name}",
        )

    try:
        scheduler.set_inner(schema.name, schema.config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return schema.dict()
