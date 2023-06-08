from httpx import Response
import respx
from PIL import Image
import torchvision.transforms as T
import numpy as np


@respx.mock
def test_modules_scenario(client) -> None:  # type: ignore
    module = "test_module"
    add_module_schema = {
        "name": module,
        "type": "triton",
        "urls": {"g4dn.xlarge": "http://g4dn.xlarge:8000", "g5.xlarge": "http://g5.xlarge:8000"},
    }

    response = client.get(f"/modules/{module}/metadata")
    assert response.status_code == 404, response.content

    # Add module with no urls
    response = client.post(f"/modules/add", json={"name": module, "type": "triton", "urls": {}})
    assert response.status_code == 422, response.content

    # Mock the response from the triton server
    dummy_module_invalid_datatype = {
        "name": module,
        "type": "triton",
        "inputs": [{"name": "input__0", "datatype": "FP32", "shape": [2, 2, 2]}],
        "outputs": [
            {"name": "output__0", "datatype": "INT64", "shape": [5]},
        ],
    }
    route_1 = respx.get(f"http://g4dn.xlarge:8000/v2/models/{module}").mock(
        return_value=Response(200, json=dummy_module_invalid_datatype)
    )
    # Add module with invalid urls
    response = client.post(f"/modules/add", json=add_module_schema)
    assert route_1.called
    assert response.status_code == 400, response.content
    assert b"Unsupported datatype" in response.content

    # Mock the response from the triton server
    dummy_module_metadata = {
        "name": module,
        "type": "triton",
        "inputs": [{"name": "input__0", "datatype": "FP32", "shape": [2, 2, 2]}],
        "outputs": [
            {"name": "output__0", "datatype": "FP32", "shape": [5]},
        ],
    }
    dummy_module_metadata_crashed = {
        "name": module,
        "type": "triton",
        "inputs": [{"name": "input__0", "datatype": "FP32", "shape": [2, 2, 2]}],
        "outputs": [
            {"name": "output__0", "datatype": "FP32", "shape": [1]},
        ],
    }
    route_1 = respx.get(f"http://g4dn.xlarge:8000/v2/models/{module}").mock(
        return_value=Response(200, json=dummy_module_metadata)
    )
    route_2 = respx.get(f"http://g5.xlarge:8000/v2/models/{module}").mock(
        return_value=Response(200, json=dummy_module_metadata_crashed)
    )
    # Add module with invalid urls
    response = client.post(f"/modules/add", json=add_module_schema)
    assert route_1.called
    assert route_2.called
    assert response.status_code == 400, response.content

    # Mock the response from the triton server
    route_1 = respx.get(f"http://g4dn.xlarge:8000/v2/models/{module}").mock(
        return_value=Response(200, json=dummy_module_metadata)
    )
    route_2 = respx.get(f"http://g5.xlarge:8000/v2/models/{module}").mock(
        return_value=Response(200, json=dummy_module_metadata)
    )
    response = client.post(f"/modules/add", json=add_module_schema)
    assert route_1.called
    assert route_2.called
    assert response.status_code == 200, response.content
    obtained_metadata = response.json()

    # Add duplicate module
    response = client.post(f"/modules/add", json=add_module_schema)
    assert response.status_code == 400, response.content

    # Get metadata
    response = client.get(f"/modules/{module}/metadata")
    assert response.status_code == 200, response.content
    assert response.json() == obtained_metadata

    response = client.get(f"/modules/list")
    assert response.status_code == 200, response.content
    assert response.json() == {module: obtained_metadata}


# TODO: change to use nms builtin module
def test_builtin_modules_scenario(client) -> None:  # type: ignore
    module = "detection_module"
    add_module_schema = {
        "name": module,
        "type": "builtin",
        "builtin_args": {"target": "FasterRCNN_ResNet50_FPN"},
    }

    # Add module
    response = client.post(f"/modules/add", json=add_module_schema)
    assert response.status_code == 200, response.content
    obtained_metadata = response.json()

    # Get metadata
    response = client.get(f"/modules/{module}/metadata")
    assert response.status_code == 200, response.content
    assert response.json() == obtained_metadata

    # Set scheduler
    scheduler_schema = {"name": "fifo", "config": {}}
    response = client.post(f"/scheduler/set", json=scheduler_schema)
    assert response.status_code == 200, response.content

    # Load input image
    img_pil = Image.open("./assets/FudanPed00054.png").convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img = transform(img_pil)
    batch_img = img.unsqueeze(0)

    infer_schema = {
        "target": add_module_schema["name"],
        "inputs": {"BATCH_IMAGE": {"data": batch_img.numpy().tolist(), "shape": batch_img.shape}},
    }
    response = client.post(f"/modules/{add_module_schema['name']}/infer", json=infer_schema)
    assert response.status_code == 200, response.content
    result = response.json()

    assert len(result) == len(obtained_metadata["outputs"])
    for output in obtained_metadata["outputs"]:
        assert output["name"] in result
        print(np.array(result[output["name"]]).shape)


# TODO: add more scenarios for testing (e.g., classification, object detection, etc.)
# For testing on k8s
def test_modules_scenario_on_k8s(client, add_module_schema) -> None:  # type: ignore
    response = client.post(f"/modules/add", json=add_module_schema)
    assert response.status_code == 200, response.content
    obtained_metadata = response.json()

    scheduler_schema = {"name": "fifo", "config": {}}
    response = client.post(f"/scheduler/set", json=scheduler_schema)
    assert response.status_code == 200, response.content

    # FIXME: generalize this test
    # NOTE: we assume we use the module from https://github.com/triton-inference-server/tutorials/tree/main/Quick_Deploy/PyTorch
    infer_schema = {
        "target": add_module_schema["name"],
        "inputs": {"input__0": {"data": [0.0] * 3 * 224 * 224}},
    }
    response = client.post(f"/modules/{add_module_schema['name']}/infer", json=infer_schema)
    assert response.status_code == 200, response.content
    result = response.json()

    assert len(result) == len(obtained_metadata["outputs"])
    for output in obtained_metadata["outputs"]:
        assert output["name"] in result
