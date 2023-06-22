from PIL import Image
import numpy as np
import requests
import torchvision.transforms as T

API_URL = "http://0.0.0.0:80"

if __name__ == "__main__":
    response = requests.delete(API_URL + "/bls/clear")
    assert response.status_code == 200, response.content
    response = requests.delete(API_URL + "/modules/clear")
    assert response.status_code == 200, response.content

    detection_module = "detection_module"
    add_module_schema = {
        "name": detection_module,
        "type": "builtin",
        "builtin_args": {"target": "FasterRCNN_ResNet50_FPN"},
    }

    response = requests.post(API_URL + "/modules/add", json=add_module_schema)
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
            "mode": "resize",
            "resize_shape": [224, 224],
        },
    }

    response = requests.post(API_URL + "/modules/add", json=add_module_schema)
    assert response.status_code == 200, response.content
    # crop_module_metadata = response.json()
    # print(crop_module_metadata)

    classifier_module = "resnet50"
    add_module_schema = {
        "name": classifier_module,
        "type": "builtin",
        "builtin_args": {"target": "Resnet50FromTorchHub"},
    }

    response = requests.post(API_URL + "/modules/add", json=add_module_schema)
    assert response.status_code == 200, response.content
    # classifier_module_metadata = response.json()
    # print(classifier_module_metadata)

    with open("../objdtc_clsfy_bls/objdtc_clsfy_bls.py", "rb") as file:
        response = requests.post(
            API_URL + "/bls/add",
            files={"file": file},
        )
    assert response.status_code == 200, response.content

    # Set scheduler
    scheduler_schema = {"name": "fifo", "config": {}}
    response = requests.post(API_URL + "/scheduler/set", json=scheduler_schema)
    assert response.status_code == 200, response.content

    # Load input image
    img_pil = Image.open("../../assets/FudanPed00054.png").convert("RGB")
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
    response = requests.post(
        API_URL + "/bls/objdtc_clsfy_bls.py/infer", json=infer_schema
    )
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

    response = requests.delete(API_URL + "/bls/clear")
    assert response.status_code == 200, response.content
    response = requests.delete(API_URL + "/modules/clear")
    assert response.status_code == 200, response.content
