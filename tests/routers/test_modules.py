import os

from PIL import Image
import cv2
from httpx import Response
import numpy as np
import respx
import torchvision.transforms as T

COCO_LABELS_LIST = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def visualize_detection_result(img_pil, boxes, labels, scores, result_path="result.png"):  # type: ignore
    """
    img_pil : pil image range - [0 255], uint8
    boxes : torch.Tensor, [num_obj, 4], torch.float32
    labels : torch.Tensor, [num_obj] torch.int64
    scores : torch.Tensor, [num_obj] torch.float32
    """

    coco_labels_map = {k: v for v, k in enumerate(COCO_LABELS_LIST)}
    np.random.seed(1)
    coco_colors_array = np.random.randint(256, size=(91, 3)) / 255

    # 1. uint8 -> float32
    image_np = np.array(img_pil).astype(np.float32) / 255.0
    x_img = image_np
    im_show = cv2.cvtColor(x_img, cv2.COLOR_RGB2BGR)

    for j in range(len(boxes)):

        label_list = list(coco_labels_map.keys())
        color_array = coco_colors_array

        x_min = int(boxes[j][0])
        y_min = int(boxes[j][1])
        x_max = int(boxes[j][2])
        y_max = int(boxes[j][3])

        cv2.rectangle(
            im_show,
            pt1=(x_min, y_min),
            pt2=(x_max, y_max),
            # FIXME: we should use int type here in the future
            color=color_array[labels[j]],
            thickness=2,
        )

        # text_size
        text_size = cv2.getTextSize(
            text=label_list[labels[j]] + " {:.2f}".format(scores[j].item()),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            thickness=1,
        )[0]

        # text_rec
        cv2.rectangle(
            im_show,
            pt1=(x_min, y_min),
            pt2=(x_min + text_size[0] + 3, y_min + text_size[1] + 4),
            color=color_array[labels[j]],
            thickness=-1,
        )

        # put text
        cv2.putText(
            im_show,
            text=label_list[labels[j]] + " {:.2f}".format(scores[j].item()),
            org=(x_min + 10, y_min + 10),  # must be int
            fontFace=0,
            fontScale=0.4,
            color=(0, 0, 0),
        )

    # cv2.imshow(...) : float values in the range [0, 1]
    # cv2.imshow("result", im_show)
    # cv2.waitKey(0)

    # cv2.imwrite(...) : int values in the range [0, 255]
    im_show = im_show * 255
    cv2.imwrite(result_path, im_show)
    return 0


def crop_person_objects(  # type: ignore
    image,
    boxes,
    labels,
    result_dir=".",
    result_name="person",
) -> int:
    person_label = COCO_LABELS_LIST.index("person")
    person_boxes = [box for box, label in zip(boxes, labels) if label == person_label]

    num_of_image = len(person_boxes)
    np_image = np.array(image)
    for i, box in enumerate(person_boxes):
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])
        cropped_np_image = np_image[y_min:y_max, x_min:x_max]
        cropped_cv_image = cv2.cvtColor(cropped_np_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(result_dir, result_name + f"_{i}" + ".png"), cropped_cv_image)

    return num_of_image


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

    # We only use the first image in the batch
    assert len(result) == len(obtained_metadata["outputs"])
    assert "BOXES" in result
    boxes = np.array(result["BOXES"])[0]
    assert "LABELS" in result
    labels = np.array(result["LABELS"])[0]
    assert "SCORES" in result
    scores = np.array(result["SCORES"])[0]

    # We only keep the boxes with scores >= 0.9
    threshold = 0.9
    indices = scores >= threshold
    pred_boxes = boxes[indices]
    pred_labels = labels[indices]
    pred_scores = scores[indices]

    visualize_detection_result(
        img_pil,
        pred_boxes,
        pred_labels,
        pred_scores,
        result_path="./assets/FudanPed00054_result.png",
    )

    print("\n>> Result is visualized at ./assets/FudanPed00054_result.png << ", end="")

    num_of_image = crop_person_objects(
        img_pil,
        pred_boxes,
        pred_labels,
        result_dir="./assets",
        result_name="FudanPed00054_person",
    )

    print("\n>> Result is cropped at ./assets/FudanPed00054_person_{ ", end="")
    for i in range(num_of_image):
        print(i, ", ", end="")
    print("} << ")


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
    # Load input image
    img_pil = Image.open("./assets/header-gulf-birds.jpg")
    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = transform(img_pil)

    infer_schema = {
        "target": add_module_schema["name"],
        "inputs": {"input__0": {"data": img.numpy().tolist()}},
    }
    response = client.post(f"/modules/{add_module_schema['name']}/infer", json=infer_schema)
    assert response.status_code == 200, response.content
    result = response.json()

    assert len(result) == len(obtained_metadata["outputs"])
    assert "output__0" in result
    array = np.array(result["output__0"]).reshape(1000)
    top5 = np.argsort(array)
    print(">> Result Top 5: [", end=" ")
    for i in top5[::-1][:5]:
        print(f"{i}: {array[i]}", end=" ")
    print("] << ", end="")
