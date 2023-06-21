from PIL import Image
import numpy as np
import pytest
import torchvision.transforms as T

from trytune.services.moduels.builtins import Resnet50FromTorchHub

# TODO: add more tests for builtin modules


@pytest.mark.asyncio
async def test_builtin_resnet50_from_torch_hub() -> None:  # type: ignore
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
    img.unsqueeze_(0)

    builtin = Resnet50FromTorchHub()
    await builtin.initialize({})
    result = await builtin.execute({"inputs": {"input__0": img.numpy()}})
    result = result["outputs"]

    assert "output__0" in result
    array = result["output__0"].reshape(1000)
    top5 = np.argsort(array)
    print(">> Result Top 5: [", end=" ")
    for i in top5[::-1][:5]:
        print(f"{i}: {array[i]}", end=" ")
    print("] << ", end="")
