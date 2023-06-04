from pydantic import BaseModel
from typing import Dict


class AddModelSchema(BaseModel):
    """
    Schema for adding model.

    Attributes:
        urls (dict): A dictionary of triton server urls for each instance type.

    Example:
        If user sends a request to add a model with the following information:
        {
            "name": "resnet50",
            "urls": {
                "g4dn.xlarge": "http://eks.ingress.url/g4dn"
                "g5.xlarge": "http://eks.ingress.url/g5"
                "inf1.xlarge": "http://eks.ingress.url/inf1"
                ...
            }
        }
        FYI, all urls are linked to the triton servers serving same model but run on different instance types

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

    Warning:
        URL should start with "http://" or "https://"

    """

    name: str
    urls: Dict[str, str]
