import pytest
import respx
from httpx import Response
from fastapi import FastAPI
from fastapi.testclient import TestClient
from trytune.routers import models  # replace 'your_application' with the name of your module


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(models.router)
    return TestClient(app)


@respx.mock
def test_model_scenario(client):
    model = "test_model"
    model_add_schema = {"urls": {"g4dn.xlarge": "http://localhost:8000"}}
    infer_schema = {
        "target": "pipe1",
        "inputs": [
            {"name": "i1", "data": [1, 2, 3]},
            {"name": "i2", "data": [4, 5, 6]},
        ],
    }
    dummy_model_metadata = {
        "name": model,
        "inputs": [{"name": "input__0", "datatype": "FP32", "shape": [3, 224, 224]}],
        "outputs": [
            {"name": "output__0", "datatype": "FP32", "shape": [1000]},
            {"name": "output__1", "datatype": "INT32", "shape": [1]},
        ],
    }

    route_1 = respx.get(f"http://localhost:8000/v2/models/{model}").mock(
        return_value=Response(200, json=dummy_model_metadata)
    )
    response = client.get(f"/models/{model}/metadata")
    assert response.status_code == 404

    response = client.post(f"/models/{model}/add", json=model_add_schema)
    assert route_1.called
    assert response.status_code == 200
    assert response.json() == dummy_model_metadata

    response = client.get(f"/models/{model}/metadata")
    assert response.status_code == 200
    assert response.json() == {"urls": model_add_schema["urls"], "metadata": dummy_model_metadata}

    # TODO: change the following to use the mock scheduler service
    # TODO: check response has two outputs with the shape of [1000] and [1]
    response = client.post(f"/models/{model}/infer", json=infer_schema)
    assert response.status_code == 200
    assert response.json() == infer_schema
