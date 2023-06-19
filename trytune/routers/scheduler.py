import traceback
from typing import Any

from fastapi import APIRouter, HTTPException

from trytune.schemas.scheduler import SetSchedulerSchema
from trytune.services.schedulers import scheduler

router = APIRouter()


@router.post("/scheduler/set")
async def set_scheduler(schema: SetSchedulerSchema) -> Any:
    try:
        await scheduler.set_inner(schema.name, schema.config)
    except Exception:
        raise HTTPException(
            status_code=400, detail=f"While setting scheduler: {traceback.format_exc()}"
        )

    return {"message": "Scheduler set", "name": schema.name, "config": schema.config}


@router.get("/scheduler/metadata")
async def get_scheduler_metadata() -> Any:
    try:
        return scheduler.get_metadata()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"While getting scheduler metadata: {traceback.format_exc()}",
        )


@router.delete("/scheduler/delete")
async def delete_scheduler() -> Any:
    try:
        await scheduler.delete_inner()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"While deleting scheduler: {traceback.format_exc()}",
        )

    return {"message": "Scheduler deleted"}
