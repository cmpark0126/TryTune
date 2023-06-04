import pytest
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from trytune.routers import models, scheduler


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()

    # To test the router, you need to include it in the app.
    app.include_router(models.router)
    app.include_router(scheduler.router)
    return TestClient(app)


@pytest.fixture
def add_model_schema() -> Dict[str, Any]:
    return {
        # FIXME: generalize this test
        # NOTE: we assume we use the model from https://github.com/triton-inference-server/tutorials/tree/main/Quick_Deploy/PyTorch
        "name": "resnet50",
        "urls": {
            "g4dn.xlarge": "http://<address>:80/path/to/server",
            "g5.xlarge": "http://<address>:80/path/to/server",
        },
    }
