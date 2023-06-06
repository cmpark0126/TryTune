from typing import Any, Dict


# Class to store module metadatas and links to triton servers.
class Modules:
    def __init__(self) -> None:
        self.modules: Dict[str, Dict[str, Any]] = {}

    def set(self, module: str, metadata: Dict[str, Any]) -> None:
        assert module not in self.modules
        self.modules[module] = metadata

    def get(self, module: str) -> Dict[str, Any]:
        return self.modules[module]


# FIXME: Avoid using singleton pattern and class variables.
# FIXME: This version of class is not thread-safe.
modules = Modules()
