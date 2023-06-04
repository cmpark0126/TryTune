from typing import Any, Dict


# Class to store model metadatas and links to triton servers.
# FIXME: Avoid using singleton pattern and class variables.
# FIXME: This version of class is not thread-safe.
class Models:
    def __init__(self) -> None:
        self.models: Dict[str, Dict[str, Any]] = {}

    def set(self, model: str, metadata: Dict[str, Any]) -> None:
        assert model not in self.models
        self.models[model] = metadata

    def get(self, model: str) -> Dict[str, Any]:
        return self.models[model]
