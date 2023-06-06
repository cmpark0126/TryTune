import respx
from httpx import Response


@respx.mock
def test_module_scenario(client) -> None:  # type: ignore
    module = "test_module"
    add_module_schema = {
        "name": module,
        "urls": {"g4dn.xlarge": "http://g4dn.xlarge:8000", "g5.xlarge": "http://g5.xlarge:8000"},
    }

    response = client.get(f"/modules/{module}/metadata")
    assert response.status_code == 404

    # Add module with no urls
    response = client.post(f"/modules/add", json={"name": module, "urls": {}})
    assert response.status_code == 400

    # Mock the response from the triton server
    dummy_module_invalid_datatype = {
        "name": module,
        "inputs": [{"name": "input__0", "datatype": "FP32", "shape": [2, 2, 2]}],
        "outputs": [
            {"name": "output__0", "datatype": "INT32", "shape": [5]},
        ],
    }
    route_1 = respx.get(f"http://g4dn.xlarge:8000/v2/modules/{module}").mock(
        return_value=Response(200, json=dummy_module_invalid_datatype)
    )
    # Add module with invalid urls
    response = client.post(f"/modules/add", json=add_module_schema)
    assert route_1.called
    assert response.status_code == 400
    assert b"Unsupported datatype" in response.content

    # Mock the response from the triton server
    dummy_module_metadata = {
        "name": module,
        "inputs": [{"name": "input__0", "datatype": "FP32", "shape": [2, 2, 2]}],
        "outputs": [
            {"name": "output__0", "datatype": "FP32", "shape": [5]},
        ],
    }
    dummy_module_metadata_crashed = {
        "name": module,
        "inputs": [{"name": "input__0", "datatype": "FP32", "shape": [2, 2, 2]}],
        "outputs": [
            {"name": "output__0", "datatype": "FP32", "shape": [1]},
        ],
    }
    route_1 = respx.get(f"http://g4dn.xlarge:8000/v2/modules/{module}").mock(
        return_value=Response(200, json=dummy_module_metadata)
    )
    route_2 = respx.get(f"http://g5.xlarge:8000/v2/modules/{module}").mock(
        return_value=Response(200, json=dummy_module_metadata_crashed)
    )
    # Add module with invalid urls
    response = client.post(f"/modules/add", json=add_module_schema)
    assert route_1.called
    assert route_2.called
    assert response.status_code == 400

    # Mock the response from the triton server
    route_1 = respx.get(f"http://g4dn.xlarge:8000/v2/modules/{module}").mock(
        return_value=Response(200, json=dummy_module_metadata)
    )
    route_2 = respx.get(f"http://g5.xlarge:8000/v2/modules/{module}").mock(
        return_value=Response(200, json=dummy_module_metadata)
    )
    response = client.post(f"/modules/add", json=add_module_schema)
    assert route_1.called
    assert route_2.called
    assert response.status_code == 200
    obtained_metadata = response.json()

    # Add duplicate module
    response = client.post(f"/modules/add", json=add_module_schema)
    assert response.status_code == 400

    # Get metadata
    response = client.get(f"/modules/{module}/metadata")
    assert response.status_code == 200
    assert response.json() == obtained_metadata

    scheduler_schema = {"name": "fifo", "config": {}}
    response = client.post(f"/scheduler/set", json=scheduler_schema)
    assert response.status_code == 200

    # TODO: test inferencing using mock server in the future
    # infer_schema = {
    #     "target": module,
    #     "inputs": {"input__0": {"data": [0.0] * 8}},  # 8 == 2 * 2 * 2
    # }
    # route_1 = respx.post(f"http://g4dn.xlarge:8001/v2/modules/{module}/infer").mock(
    #     return_value=Response(200, json=dummy_result)
    # )
    # response = client.post(f"/modules/infer", json=infer_schema)
    # assert route_1.called
    # assert response.status_code == 200
    # result = response.json()
    # assert len(result) == 1
    # assert "output__1" in result
    # assert len(result["output__0"].data) == 5


# For testing on k8s
def test_module_scenario_on_k8s(client, add_module_schema) -> None:  # type: ignore
    response = client.post(f"/modules/add", json=add_module_schema)
    assert response.status_code == 200
    obtained_metadata = response.json()

    scheduler_schema = {"name": "fifo", "config": {}}
    response = client.post(f"/scheduler/set", json=scheduler_schema)
    assert response.status_code == 200

    # FIXME: generalize this test
    # NOTE: we assume we use the module from https://github.com/triton-inference-server/tutorials/tree/main/Quick_Deploy/PyTorch
    infer_schema = {
        "target": add_module_schema["name"],
        "inputs": {"input__0": {"data": [0.0] * 3 * 224 * 224}},
    }
    response = client.post(f"/modules/infer", json=infer_schema)
    assert response.status_code == 200
    result = response.json()

    assert len(result) == len(obtained_metadata["outputs"])
    for output in obtained_metadata["outputs"]:
        assert output["name"] in result
