from PIL import Image
import numpy as np
import torchvision.transforms as T


def test_bls_scenario(client) -> None:  # type: ignore
    detection_module = "detection_module"
    add_module_schema = {
        "name": detection_module,
        "type": "builtin",
        "builtin_args": {"target": "FasterRCNN_ResNet50_FPN"},
    }

    response = client.post("/modules/add", json=add_module_schema)
    assert response.status_code == 200, response.content
    # detection_module_metadata = response.json()
    # print(detection_module_metadata)

    crop_module = "crop_module"
    add_module_schema = {
        "name": crop_module,
        "type": "builtin",
        "builtin_args": {
            "target": "Crop",
            "threshold": 0.9,
            "label": 1,  # label 1 is person
        },
    }

    response = client.post("/modules/add", json=add_module_schema)
    assert response.status_code == 200, response.content
    # crop_module_metadata = response.json()
    # print(crop_module_metadata)

    classifier_module = "classifier_module"
    add_module_schema = {
        "name": classifier_module,
        "type": "builtin",
        "builtin_args": {"target": "Resnet50FromTorchHub"},
    }

    response = client.post("/modules/add", json=add_module_schema)
    assert response.status_code == 200, response.content
    # classifier_module_metadata = response.json()
    # print(classifier_module_metadata)

    with open("./examples/objdtc_clsfy_bls/objdtc_clsfy_bls.py", "rb") as file:
        response = client.post(
            "/bls/add",
            files={"file": file},
        )
    assert response.status_code == 200, response.content

    # Set scheduler
    scheduler_schema = {"name": "fifo", "config": {}}
    response = client.post("/scheduler/set", json=scheduler_schema)
    assert response.status_code == 200, response.content

    # Load input image
    img_pil = Image.open("./assets/FudanPed00054.png").convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img = transform(img_pil)
    batch_img = img.unsqueeze(0)

    infer_schema = {
        "target": "objdtc_clsfy_bls.py",
        "inputs": {
            "p_image": {
                "data": batch_img.numpy().tolist(),
                "shape": batch_img.shape,
            }
        },
    }
    response = client.post("/bls/objdtc_clsfy_bls.py/infer", json=infer_schema)
    assert response.status_code == 200, response.content
    result = response.json()
    assert "p_output__0" in result
    classification_results = result["p_output__0"]
    for result in classification_results:
        array = np.array(result).reshape(1000)
        top5 = np.argsort(array)
        print(">> Result Top 5: [", end=" ")
        for i in top5[::-1][:5]:
            print(f"{i}: {array[i]}", end=" ")
        print("] << ")


# TODO: add more scenarios for testing (e.g., classification, object detection, etc.)
# For testing on k8s
def test_bls_scenario_on_k8s(client, add_module_schemas, add_pipeline_schema) -> None:  # type: ignore
    pass
