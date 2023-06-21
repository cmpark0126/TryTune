from PIL import Image
from locust import FastHttpUser, TaskSet, between, events, task
import requests
import torchvision.transforms as T

YOUR_API_URL = "http://0.0.0.0:80"


@events.test_start.add_listener
def on_test_start(**kw):  # type: ignore
    print("test is starting")
    response = requests.delete(YOUR_API_URL + "/bls/clear")
    assert response.status_code == 200, response.content
    response = requests.delete(YOUR_API_URL + "/modules/clear")
    assert response.status_code == 200, response.content

    detection_module = "detection_module"
    add_module_schema = {
        "name": detection_module,
        "type": "builtin",
        "builtin_args": {"target": "FasterRCNN_ResNet50_FPN"},
    }

    response = requests.post(YOUR_API_URL + "/modules/add", json=add_module_schema)
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

    response = requests.post(YOUR_API_URL + "/modules/add", json=add_module_schema)
    assert response.status_code == 200, response.content
    # crop_module_metadata = response.json()
    # print(crop_module_metadata)

    classifier_module = "classifier_module"
    add_module_schema = {
        "name": classifier_module,
        "type": "builtin",
        "builtin_args": {"target": "Resnet50FromTorchHub"},
    }

    response = requests.post(YOUR_API_URL + "/modules/add", json=add_module_schema)
    assert response.status_code == 200, response.content
    # classifier_module_metadata = response.json()
    # print(classifier_module_metadata)

    with open("../objdtc_clsfy_bls/objdtc_clsfy_bls.py", "rb") as file:
        response = requests.post(
            YOUR_API_URL + "/bls/add",
            files={"file": file},
        )
    assert response.status_code == 200, response.content

    # Set scheduler
    scheduler_schema = {"name": "fifo", "config": {}}
    response = requests.post(YOUR_API_URL + "/scheduler/set", json=scheduler_schema)
    assert response.status_code == 200, response.content


@events.test_stop.add_listener
def on_test_stop(**kw):  # type: ignore
    response = requests.delete(YOUR_API_URL + "/bls/clear")
    assert response.status_code == 200, response.content
    response = requests.delete(YOUR_API_URL + "/modules/clear")
    assert response.status_code == 200, response.content


class UserBehavior(TaskSet):
    def on_start(self):  # type: ignore
        print("nested on start")
        # Load input image
        img_pil = Image.open("../../assets/FudanPed00054.png").convert("RGB")
        transform = T.Compose([T.ToTensor()])
        img = transform(img_pil)
        batch_img = img.unsqueeze(0)

        self.infer_schema = {
            "target": "objdtc_clsfy_bls.py",
            "inputs": {
                "p_image": {
                    "data": batch_img.numpy().tolist(),
                    "shape": batch_img.shape,
                }
            },
        }

    def on_stop(self):  # type: ignore
        print("nested on stop")

    @task(1)
    def infer(self):  # type: ignore
        response = self.client.post(
            "/bls/objdtc_clsfy_bls.py/infer", json=self.infer_schema
        )
        assert response.status_code == 200, response.content
        result = response.json()
        assert "p_output__0" in result


class User(FastHttpUser):
    tasks = {UserBehavior: 1}  # type: ignore
    wait_time = between(0.1, 0.2)
