import asyncio
from typing import Any

import numpy as np

from trytune.routers.common import infer_module, infer_module_with_async_queue


async def execute(tensors: Any) -> Any:
    # detection
    detection_module = {
        "name": "detector",
        "module": "detection_module",
        "inputs": {"BATCH_IMAGE": {"name": "p_image"}},
        "outputs": {
            "BOXES": {"name": "p_boxes"},
            "LABELS": {"name": "p_labels"},
            "SCORES": {"name": "p_scores"},
        },
    }
    inputs = {}
    for src, dst in detection_module["inputs"].items():  # type: ignore
        data = tensors[dst["name"]]
        inputs[src] = data

    outputs = await infer_module(detection_module["module"], inputs)  # type: ignore

    for src, dst in detection_module["outputs"].items():  # type: ignore
        assert dst["name"] not in tensors
        data = outputs[src]
        tensors[dst["name"]] = outputs[src]

    # crop
    crop_module = {
        "name": "cropper",
        "module": "crop_module",
        "inputs": {
            "IMAGE": {"name": "p_image"},
            "BOXES": {"name": "p_boxes"},
            "LABELS": {"name": "p_labels"},
            "SCORES": {"name": "p_scores"},
        },
        "outputs": {
            "CROPPED_IMAGES": {"name": "p_cropped_images"},
            "WHS": {"name": "p_whs"},
        },
    }
    inputs = {}
    for src, dst in crop_module["inputs"].items():  # type: ignore
        data = tensors[dst["name"]]
        inputs[src] = data

    outputs = await infer_module(crop_module["module"], inputs)  # type: ignore

    for src, dst in crop_module["outputs"].items():  # type: ignore
        assert dst["name"] not in tensors
        data = outputs[src]
        tensors[dst["name"]] = outputs[src]

    # NOTE: parallel execution
    # classifier
    cropped_images = tensors["p_cropped_images"]
    cropped_images = np.split(cropped_images, cropped_images.shape[0], axis=0)

    queue = asyncio.Queue()  # type: ignore

    # NOTE: parallel execution with async queue
    num_of_cropped_images = len(cropped_images)
    for i, cropped_image in enumerate(cropped_images):
        cropped_image = cropped_image.reshape(cropped_image.shape[1:])
        inputs = {"input__0": cropped_image}
        output_map = {"output__0": {"name": f"output__0_{i}"}}
        asyncio.create_task(
            infer_module_with_async_queue("resnet50", inputs, output_map, queue)
        )

    clsfy_results = []
    for _ in range(num_of_cropped_images):
        event = await queue.get()
        if "error" in event:
            raise event["error"]
        # print(event["name"])
        clsfy_results.append(event["tensor"].tolist())

    response = {"p_output__0": clsfy_results}
    return response
