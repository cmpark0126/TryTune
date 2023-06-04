from typing import Any, Dict


# Class to store model metadatas and links to triton servers.
# FIXME: Avoid using singleton pattern and class variables.
# FIXME: This version of class is not thread-safe.
class ModelRegistry:
    def __init__(self) -> None:
        self.models: Dict[str, Dict[str, Any]] = {}

    def add(self, model: str, metadata: Dict[str, Any]) -> None:
        self.models[model] = metadata

    def get_metadata(self, model: str) -> Dict[str, Any]:
        return self.models[model]
