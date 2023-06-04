from fastapi import FastAPI
from trytune.routers import pipelines, models

app = FastAPI()

app.include_router(pipelines.router)
app.include_router(models.router)
