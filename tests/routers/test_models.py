import pytest
import respx
from httpx import Response
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


@respx.mock
def test_model_scenario(client) -> None:  # type: ignore
    model = "test_model"
    model_add_schema = {
        "name": model,
        "urls": {"g4dn.xlarge": "http://g4dn.xlarge:8000", "g5.xlarge": "http://g5.xlarge:8000"},
    }

    response = client.get(f"/models/{model}/metadata")
    assert response.status_code == 404

    # Add model with no urls
    response = client.post(f"/models/add", json={"name": model, "urls": {}})
    assert response.status_code == 400

    # Mock the response from the triton server
    dummy_model_invalid_datatype = {
        "name": model,
        "inputs": [{"name": "input__0", "datatype": "FP32", "shape": [2, 2, 2]}],
        "outputs": [
            {"name": "output__0", "datatype": "INT32", "shape": [5]},
        ],
    }
    route_1 = respx.get(f"http://g4dn.xlarge:8000/v2/models/{model}").mock(
        return_value=Response(200, json=dummy_model_invalid_datatype)
    )
    # Add model with invalid urls
    response = client.post(f"/models/add", json=model_add_schema)
    assert route_1.called
    assert response.status_code == 400
    assert b"Unsupported datatype" in response.content

    # Mock the response from the triton server
    dummy_model_metadata = {
        "name": model,
        "inputs": [{"name": "input__0", "datatype": "FP32", "shape": [2, 2, 2]}],
        "outputs": [
            {"name": "output__0", "datatype": "FP32", "shape": [5]},
        ],
    }
    dummy_model_metadata_crashed = {
        "name": model,
        "inputs": [{"name": "input__0", "datatype": "FP32", "shape": [2, 2, 2]}],
        "outputs": [
            {"name": "output__0", "datatype": "FP32", "shape": [1]},
        ],
    }
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
    route_1 = respx.get(f"http://g4dn.xlarge:8000/v2/models/{model}").mock(
        return_value=Response(200, json=dummy_model_metadata)
    )
    route_2 = respx.get(f"http://g5.xlarge:8000/v2/models/{model}").mock(
        return_value=Response(200, json=dummy_model_metadata)
    )
    response = client.post(f"/models/add", json=model_add_schema)
    assert route_1.called
    assert route_2.called
    assert response.status_code == 200
    obtained_metadata = response.json()

    # Add duplicate model
    response = client.post(f"/models/add", json=model_add_schema)
    assert response.status_code == 400

    # Get metadata
    response = client.get(f"/models/{model}/metadata")
    assert response.status_code == 200
    assert response.json() == obtained_metadata

    scheduler_schema = {"name": "fifo", "config": {}}
    response = client.post(f"/scheduler/set", json=scheduler_schema)
    assert response.status_code == 200

    # infer_schema = {
    #     "target": model,
    #     "inputs": {
    #         "i1": {"data": [1.0, 2.0, 3.0]},
    #         "i2": {"data": [4.0, 5.0, 6.0]},
    #     },
    # }
    # route_7 = respx.post(f"http://g5.xlarge:8001/v2/models/{model}/infer").mock(
    #     return_value=Response(200, json=dummy_model_metadata)
    # )
    # response = client.post(f"/models/infer", json=infer_schema)
    # assert route_7.called
    # assert response.status_code == 200
    # result = response.json()
    # assert len(result) == 2
    # if result[0]["name"] == "output__0":
    #     assert result[1]["name"] == "output__1"
    #     assert len(result[0]["data"]) == 1000
    #     assert len(result[1]["data"]) == 1
    # elif result[1]["name"] == "output__0":
    #     assert result[0]["name"] == "output__1"
    #     assert len(result[1]["data"]) == 1000
    #     assert len(result[0]["data"]) == 1
    # else:
    #     assert False
