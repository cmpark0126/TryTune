import importlib.util
import os
import shutil
import tempfile
from typing import Any, Dict

from fastapi import APIRouter, File, HTTPException, UploadFile
import numpy as np

from trytune.schemas import common

router = APIRouter()


class TempDir:
    def __init__(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        print(f"Created temp dir: {self.temp_dir}")

    def get_path(self) -> str:
        return self.temp_dir

    def cleanup(self) -> None:
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Removed temp dir: {self.temp_dir}")


temp_dir = TempDir()


@router.get("/bls/list")
async def get_list() -> Any:
    data = []
    for filename in os.listdir(temp_dir.get_path()):
        if filename.endswith(".py"):
            data.append(filename)
    return data


@router.delete("/bls/clear")
async def clear() -> Any:
    os.removedirs(temp_dir.get_path())
    return {"message": "All Bls cleared"}


@router.post("/bls/add")
async def add_bls(file: UploadFile = File(...)) -> Any:
    contents = await file.read()
    # FIXME: check file is a valid bls file
    assert type(contents) == bytes

    # FIXME: avoid magic string
    os.makedirs(temp_dir.get_path(), exist_ok=True)

    path = os.path.join(temp_dir.get_path(), file.filename)
    if os.path.exists(path):
        raise HTTPException(
            status_code=400, detail=f"Bls {file.filename} already exists."
        )
    with open(path, "wb") as f:
        f.write(contents)

    # Return the response with the stored information
    return {"message": f"Bls {file.filename} added"}


# NOTE: Scheduler still needs pipeline dag structure to schedule
@router.post("/bls/{bls}/infer")
async def infer(bls: str, schema: common.InferSchema) -> Any:
    if bls != schema.target:
        raise HTTPException(
            status_code=400, detail=f"Bls {bls} does not match target {schema.target}"
        )

    tensors: Dict[str, Any] = {}
    for name, input in schema.inputs.items():
        tensors[name] = np.array(input.data)

    filepath = os.path.join(temp_dir.get_path(), bls)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    spec = importlib.util.spec_from_file_location("module.name", filepath)
    # FIXME: more specific error
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    if not hasattr(module, "execute"):
        raise HTTPException(status_code=404, detail="Function not found")

    execute = getattr(module, "execute")
    return await execute(tensors)
