from fastapi import APIRouter, HTTPException
from typing import Any
from trytune.schemas.scheduler import SetSchedulerSchema
from trytune.services.scheduler import Scheduler

router = APIRouter()
scheduler = Scheduler()


@router.post("/scheduler/set")
async def set_scheduler(schema: SetSchedulerSchema) -> Any:
    try:
        await scheduler.set_inner(schema.name, schema.config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"message": "Scheduler set", "name": schema.name, "config": schema.config}


@router.get("/scheduler/metadata")
async def get_scheduler_metadata() -> Any:
    try:
        return scheduler.get_metadata()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/scheduler/delete")
async def delete_scheduler() -> Any:
    try:
        await scheduler.delete_inner()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"message": "Scheduler deleted"}
