import respx
from httpx import Response


@respx.mock
def test_model_scenario(client) -> None:  # type: ignore
    model = "test_model"
    add_model_schema = {
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
    response = client.post(f"/models/add", json=add_model_schema)
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
    response = client.post(f"/models/add", json=add_model_schema)
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
    response = client.post(f"/models/add", json=add_model_schema)
    assert route_1.called
    assert route_2.called
    assert response.status_code == 200
    obtained_metadata = response.json()

    # Add duplicate model
    response = client.post(f"/models/add", json=add_model_schema)
    assert response.status_code == 400

    # Get metadata
    response = client.get(f"/models/{model}/metadata")
    assert response.status_code == 200
    assert response.json() == obtained_metadata

    scheduler_schema = {"name": "fifo", "config": {}}
    response = client.post(f"/scheduler/set", json=scheduler_schema)
    assert response.status_code == 200

    # TODO: test inferencing using mock server in the future
    # infer_schema = {
    #     "target": model,
    #     "inputs": {"input__0": {"data": [0.0] * 8}},  # 8 == 2 * 2 * 2
    # }
    # route_1 = respx.post(f"http://g4dn.xlarge:8001/v2/models/{model}/infer").mock(
    #     return_value=Response(200, json=dummy_result)
    # )
    # response = client.post(f"/models/infer", json=infer_schema)
    # assert route_1.called
    # assert response.status_code == 200
    # result = response.json()
    # assert len(result) == 1
    # assert "output__1" in result
    # assert len(result["output__0"].data) == 5


# For testing on k8s
def test_model_scenario_on_k8s(client, add_model_schema) -> None:  # type: ignore
    response = client.post(f"/models/add", json=add_model_schema)
    assert response.status_code == 200
    obtained_metadata = response.json()

    scheduler_schema = {"name": "fifo", "config": {}}
    response = client.post(f"/scheduler/set", json=scheduler_schema)
    assert response.status_code == 200

    # FIXME: generalize this test
    # NOTE: we assume we use the model from https://github.com/triton-inference-server/tutorials/tree/main/Quick_Deploy/PyTorch
    infer_schema = {
        "target": add_model_schema["name"],
        "inputs": {"input__0": {"data": [0.0] * 3 * 224 * 224}},
    }
    response = client.post(f"/models/infer", json=infer_schema)
    assert response.status_code == 200
    result = response.json()

    assert len(result) == len(obtained_metadata["outputs"])
    for output in obtained_metadata["outputs"]:
        assert output["name"] in result
