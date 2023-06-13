def test_pipelines_scenario_scenario(client) -> None:  # type: ignore
    detection_module = "detection_module"
    add_module_schema = {
        "name": detection_module,
        "type": "builtin",
        "builtin_args": {"target": "FasterRCNN_ResNet50_FPN"},
    }

    response = client.post("/modules/add", json=add_module_schema)
    assert response.status_code == 200, response.content
    detection_module_metadata = response.json()
    print(detection_module_metadata)

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
    crop_module_metadata = response.json()
    print(crop_module_metadata)

    # pipeline = "test_pipeline"
    # add_pipeline_schema = {
    #     "name": pipeline,
    #     "tensors": {
    #         "inputs": [{"name": "batch_image"}],
    #         "outputs": [{"name": "cropped_images"}],
    #         "interms": [{"name": "boxes"}, {"name": "labels"}, {"name": "scores"}],
    #     },
    #     "stages": [
    #         {
    #             "name": "classifier",
    #             "module": "resnet50",
    #             "inputs": [{"src": "input__0", "tgt": "pinput__0"}],
    #             "outputs": [{"src": "output__0", "tgt": "pinterm__0"}],
    #         },
    #         {
    #             "name": "selector",
    #             "module": "top_five",
    #             "inputs": [{"src": "input__0", "tgt": "pinterm__0"}],
    #             "outputs": [{"src": "output__0", "tgt": "poutput__0"}],
    #         },
    #     ],
    # }

    raise NotImplementedError


# TODO: add more scenarios for testing (e.g., classification, object detection, etc.)
# For testing on k8s
def test_pipelines_scenario_scenario_on_k8s(client, add_module_schemas, add_pipeline_schema) -> None:  # type: ignore
    pass
