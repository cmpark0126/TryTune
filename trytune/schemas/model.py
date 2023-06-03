from pydantic import BaseModel
from typing import Dict


class ModelAddSchema(BaseModel):
    """
    Schema for adding model.

    Attributes:
        linkes (dict): A dictionary of triton server links for each instance type.

    Example:
        If user sends a request to add a model with the following information:
        {
            # all links are linked to the triton servers serving same model
            # but run on different instance types
            "linkes": {
                "g4dn.xlarge": "eks.ingress.url/g4dn"
                "g5.xlarge": "eks.ingress.url/g5"
                "inf1.xlarge": "eks.ingress.url/inf1"
                ...
            }
        }

        Then the model registry will store the following information
        obtained from the triton server:
        {
            "name": "resnet50",
            "inputs": [
                {"name": "input__0", "dtype": "FP32", "shape": [3, 224, 224]}
            ],
            "outputs": [
                {"name": "output__0", "dtype": "FP32", "shape": [1000]}
            ],
        }

        Finnaly, send the above information back to the user also.

    """

    linkes: Dict[str, str]
