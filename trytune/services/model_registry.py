from typing import Any, Dict


# Class to store model metadatas and links to triton servers.
# FIXME: Avoid using singleton pattern and class variables.
# FIXME: This version of class is not thread-safe.
class ModelRegistry:
    models: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def add(model: str, metadata: Dict[str, Any]) -> None:
        ModelRegistry.models[model] = metadata

    @staticmethod
    def get_metadata(model: str) -> Dict[str, Any]:
        return ModelRegistry.models[model]
