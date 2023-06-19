from fastapi import FastAPI

from trytune.routers import bls, modules, pipelines, scheduler

app = FastAPI()

app.include_router(pipelines.router)
app.include_router(modules.router)
app.include_router(scheduler.router)
app.include_router(bls.router)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    print("Shutting down...")
    bls.temp_dir.cleanup()
