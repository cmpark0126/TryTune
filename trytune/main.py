from fastapi import FastAPI
from trytune.routers import modules, pipelines

app = FastAPI()

app.include_router(pipelines.router)
app.include_router(modules.router)
