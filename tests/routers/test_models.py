import pytest
import respx
from httpx import Response
from fastapi import FastAPI
from fastapi.testclient import TestClient
from trytune.routers import models


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()

    # To test the router, you need to include it in the app.
    app.include_router(models.router)
    return TestClient(app)


@respx.mock
def test_model_scenario(client) -> None:  # type: ignore
    model = "test_model"
    model_add_schema = {
        "name": model,
        "urls": {"g4dn.xlarge": "http://g4dn.xlarge:8000", "g5.xlarge": "http://g5.xlarge:8000"},
    }
    model_add_schema_2 = {
        "name": model,
        "urls": {"g4dn.xlarge": "http://g4dn.xlarge:8001", "g5.xlarge": "http://g5.xlarge:8001"},
    }
    infer_schema = {
        "target": model,
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
    dummy_model_metadata_crashed = {
        "name": model,
        "inputs": [{"name": "input__0", "datatype": "FP32", "shape": [3, 224, 224]}],
        "outputs": [
            {"name": "output__0", "datatype": "FP32", "shape": [1000]},
        ],
    }

    response = client.get(f"/models/{model}/metadata")
    assert response.status_code == 404

    # Add model with no urls
    response = client.post(f"/models/add", json={"name": model, "urls": {}})
    assert response.status_code == 400

    # Mock the response from the triton server
    route_1 = respx.get(f"http://g4dn.xlarge:8000/v2/models/{model}").mock(
        return_value=Response(200, json=dummy_model_metadata)
    )
    route_2 = respx.get(f"http://g5.xlarge:8000/v2/models/{model}").mock(
        return_value=Response(200, json=dummy_model_metadata_crashed)
    )
    # Add model with invalid urls
    response = client.post(f"/models/add", json=model_add_schema)
    assert route_1.called
    assert route_2.called
    assert response.status_code == 400

    # Mock the response from the triton server
    route_3 = respx.get(f"http://g4dn.xlarge:8001/v2/models/{model}").mock(
        return_value=Response(200, json=dummy_model_metadata)
    )
    route_4 = respx.get(f"http://g5.xlarge:8001/v2/models/{model}").mock(
        return_value=Response(200, json=dummy_model_metadata)
    )
    response = client.post(f"/models/add", json=model_add_schema_2)
    assert route_3.called
    assert route_4.called
    assert response.status_code == 200
    dummy_model_metadata["urls"] = model_add_schema_2["urls"]  # type: ignore
    obtained_metadata = response.json()

    # Add duplicate model
    response = client.post(f"/models/add", json=model_add_schema)
    assert response.status_code == 400

    # Get metadata
    response = client.get(f"/models/{model}/metadata")
    assert response.status_code == 200
    assert response.json() == obtained_metadata

    # TODO: change the following to use the mock scheduler service
    # TODO: check the behavior when the input data is not valid
    # TODO: check response has two outputs with the shape of [1000] and [1]
    # response = client.post(f"/models/infer", json=infer_schema)
    # assert response.status_code == 200
    # assert response.json() == infer_schema
