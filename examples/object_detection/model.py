import json
import torch
import numpy as np
from typing import Dict, Any
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

import triton_python_backend_utils as pb_utils  # type: ignore


class TritonPythonModel:
    def initialize(self, args):  # type: ignore
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        boxes_config = pb_utils.get_output_config_by_name(model_config, "BOXES")
        self.boxes_dtype = pb_utils.triton_string_to_numpy(boxes_config["data_type"])

        labels_config = pb_utils.get_output_config_by_name(model_config, "LABELS")
        self.labels_dtype = pb_utils.triton_string_to_numpy(labels_config["data_type"])

        scores_config = pb_utils.get_output_config_by_name(model_config, "SCORES")
        self.scores_dtype = pb_utils.triton_string_to_numpy(scores_config["data_type"])

        # Instantiate the PyTorch model
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights, progress=False)
        self.model.eval()

    def execute(self, requests):  # type: ignore
        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            batch_image = torch.from_numpy(
                pb_utils.get_input_tensor_by_name(request, "BATCh_IMAGE").as_numpy()
            )

            # NOTE: batch_image must be B x C x H x W shapes
            print(batch_image.shape)
            preds = self.model(batch_image)
            # preds: [{"boxes": <torch.Size([41, 4])>, "labels": torch.Size([41]), "scores": torch.Size([41])}, ...]
            batch_boxes = []
            batch_labels = []
            batch_scores = []
            for pred in preds:
                # pred: {"boxes": <torch.Size([41, 4])>, "labels": torch.Size([41]), "scores": torch.Size([41])}
                batch_boxes.append(pred["boxes"].detach().numpy())
                batch_labels.append(pred["labels"].detach().numpy())
                batch_scores.append(pred["scores"].detach().numpy())

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            # FIXME: Optimize in the future to avoid unnecessary copy
            batch_boxes_np = np.stack(batch_boxes)
            batch_boxes_pb = pb_utils.Tensor("BOXES", batch_boxes_np.astype(self.boxes_dtype))

            batch_labels_np = np.stack(batch_labels)
            batch_labels_pb = pb_utils.Tensor("LABELS", batch_labels_np.astype(self.labels_dtype))

            batch_scores_np = np.stack(batch_scores)
            batch_scores_pb = pb_utils.Tensor("SCORES", batch_scores_np.astype(self.scores_dtype))

            # Create InferenceResponse.
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[batch_boxes_pb, batch_labels_pb, batch_scores_pb]
            )
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):  # type: ignore
        print("Cleaning up...")


# NOTE: if you want to test this model locally, you can use the following code and comment out `import triton_python_backend_utils as pb_utils` too
# # To simulate triton python backend code without installing triton
# import types

# pb_utils = types.ModuleType("pb_utils")


# def get_output_config_by_name(model_config: Dict[str, Any], name: str) -> Any:
#     for output in model_config["outputs"]:
#         if output["name"] == name:
#             return output
#     raise ValueError(f"Output {name} not found")


# pb_utils.get_output_config_by_name = get_output_config_by_name  # type: ignore


# def triton_string_to_numpy(  # type: ignore
#     triton_dtype: str,
# ) -> Any:
#     if triton_dtype == "FP32":
#         return np.float32
#     else:
#         raise ValueError(f"Unknown Triton dtype {triton_dtype}")


# pb_utils.triton_string_to_numpy = triton_string_to_numpy  # type: ignore


# class Tensor:
#     def __init__(self, name: str, inner: np.ndarray) -> None:
#         self.name = name
#         self.inner = inner

#     def as_numpy(self) -> np.ndarray:
#         return self.inner


# pb_utils.Tensor = Tensor  # type: ignore


# def get_input_tensor_by_name(request: Dict[str, Any], name: str) -> pb_utils.Tensor:  # type: ignore
#     for input in request["inputs"]:
#         if input.name == name:
#             return input
#     raise ValueError(f"Input {name} not found")


# pb_utils.get_input_tensor_by_name = get_input_tensor_by_name  # type: ignore


# class InferenceResponse:
#     def __init__(self, output_tensors: Any) -> None:
#         self.output_tensors = output_tensors

#     def get_output_tensor_by_name(self, name: str) -> pb_utils.Tensor:  # type: ignore
#         for output in self.output_tensors:
#             if output.name == name:
#                 return output
#         raise ValueError(f"Output {name} not found")


# pb_utils.InferenceResponse = InferenceResponse  # type: ignore


# def visualize_detection_result(img_pil, boxes, labels, scores, result_path="result.png"):  # type: ignore
#     """
#     img_pil : pil image range - [0 255], uint8
#     boxes : torch.Tensor, [num_obj, 4], torch.float32
#     labels : torch.Tensor, [num_obj] torch.int64
#     scores : torch.Tensor, [num_obj] torch.float32
#     """
#     import cv2

#     coco_labels_list = [
#         "__background__",
#         "person",
#         "bicycle",
#         "car",
#         "motorcycle",
#         "airplane",
#         "bus",
#         "train",
#         "truck",
#         "boat",
#         "traffic light",
#         "fire hydrant",
#         "N/A",
#         "stop sign",
#         "parking meter",
#         "bench",
#         "bird",
#         "cat",
#         "dog",
#         "horse",
#         "sheep",
#         "cow",
#         "elephant",
#         "bear",
#         "zebra",
#         "giraffe",
#         "N/A",
#         "backpack",
#         "umbrella",
#         "N/A",
#         "N/A",
#         "handbag",
#         "tie",
#         "suitcase",
#         "frisbee",
#         "skis",
#         "snowboard",
#         "sports ball",
#         "kite",
#         "baseball bat",
#         "baseball glove",
#         "skateboard",
#         "surfboard",
#         "tennis racket",
#         "bottle",
#         "N/A",
#         "wine glass",
#         "cup",
#         "fork",
#         "knife",
#         "spoon",
#         "bowl",
#         "banana",
#         "apple",
#         "sandwich",
#         "orange",
#         "broccoli",
#         "carrot",
#         "hot dog",
#         "pizza",
#         "donut",
#         "cake",
#         "chair",
#         "couch",
#         "potted plant",
#         "bed",
#         "N/A",
#         "dining table",
#         "N/A",
#         "N/A",
#         "toilet",
#         "N/A",
#         "tv",
#         "laptop",
#         "mouse",
#         "remote",
#         "keyboard",
#         "cell phone",
#         "microwave",
#         "oven",
#         "toaster",
#         "sink",
#         "refrigerator",
#         "N/A",
#         "book",
#         "clock",
#         "vase",
#         "scissors",
#         "teddy bear",
#         "hair drier",
#         "toothbrush",
#     ]
#     coco_labels_map = {k: v for v, k in enumerate(coco_labels_list)}
#     np.random.seed(1)
#     coco_colors_array = np.random.randint(256, size=(91, 3)) / 255

#     # 1. uint8 -> float32
#     image_np = np.array(img_pil).astype(np.float32) / 255.0
#     x_img = image_np
#     im_show = cv2.cvtColor(x_img, cv2.COLOR_RGB2BGR)

#     for j in range(len(boxes)):

#         label_list = list(coco_labels_map.keys())
#         color_array = coco_colors_array

#         x_min = int(boxes[j][0])
#         y_min = int(boxes[j][1])
#         x_max = int(boxes[j][2])
#         y_max = int(boxes[j][3])

#         cv2.rectangle(
#             im_show,
#             pt1=(x_min, y_min),
#             pt2=(x_max, y_max),
#             # FIXME: we should use int type here in the future
#             color=color_array[int(labels[j])],
#             thickness=2,
#         )

#         # text_size
#         text_size = cv2.getTextSize(
#             text=label_list[int(labels[j])] + " {:.2f}".format(scores[j].item()),
#             fontFace=cv2.FONT_HERSHEY_PLAIN,
#             fontScale=1,
#             thickness=1,
#         )[0]

#         # text_rec
#         cv2.rectangle(
#             im_show,
#             pt1=(x_min, y_min),
#             pt2=(x_min + text_size[0] + 3, y_min + text_size[1] + 4),
#             color=color_array[int(labels[j])],
#             thickness=-1,
#         )

#         # put text
#         cv2.putText(
#             im_show,
#             text=label_list[int(labels[j])] + " {:.2f}".format(scores[j].item()),
#             org=(x_min + 10, y_min + 10),  # must be int
#             fontFace=0,
#             fontScale=0.4,
#             color=(0, 0, 0),
#         )

#     # cv2.imshow(...) : float values in the range [0, 1]
#     # cv2.imshow("result", im_show)
#     # cv2.waitKey(0)

#     # cv2.imwrite(...) : int values in the range [0, 255]
#     im_show = im_show * 255
#     cv2.imwrite(result_path, im_show)
#     return 0


# if __name__ == "__main__":
#     import numpy as np
#     from PIL import Image
#     import torchvision.transforms as T
#     import argparse

#     parser = argparse.ArgumentParser(description="Test Triton Python Backend Model")
#     parser.add_argument("--input", type=str, help="Input file path")
#     parser.add_argument("--output", type=str, help="Output file path")
#     args = parser.parse_args()

#     tmp = TritonPythonModel()

#     # Initialize the model
#     # FIXME: we should use int type for labels in the future
#     json_data = '{"outputs": [{"name": "BOXES", "data_type": "FP32"},{"name": "LABELS", "data_type": "FP32"},{"name": "SCORES", "data_type": "FP32"}]}'
#     tmp.initialize({"model_config": json_data})

#     # We only has one request and response here
#     img_pil = Image.open(args.input).convert("RGB")
#     transform = T.Compose([T.ToTensor()])
#     img = transform(img_pil)
#     # Since we only have one image, we need to add a batch dimension
#     batch_img = img.unsqueeze(0)
#     tensor = pb_utils.Tensor("BATCh_IMAGE", batch_img.numpy())
#     request = {"inputs": [tensor]}
#     response = tmp.execute([request])[0]

#     # Validate the output tensor. We only have single batch here
#     boxes = response.get_output_tensor_by_name("BOXES").as_numpy()[0]
#     print(boxes.shape)
#     labels = response.get_output_tensor_by_name("LABELS").as_numpy()[0]
#     print(labels.shape)
#     scores = response.get_output_tensor_by_name("SCORES").as_numpy()[0]
#     print(scores.shape)

#     # We only keep the boxes with scores >= 0.9
#     threshold = 0.9
#     indices = scores >= threshold
#     pred_boxes = boxes[indices]
#     pred_labels = labels[indices]
#     pred_scores = scores[indices]

#     visualize_detection_result(
#         img_pil, pred_boxes, pred_labels, pred_scores, result_path=args.output
#     )
