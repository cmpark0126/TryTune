from abc import ABC, abstractmethod
from typing import Any, Dict

from trytune.schemas.common import DataSchema


class BuiltinModule(ABC):
    """
    BuiltinModule is a base class for all builtin modules. Builtin modules are executed in API server.
    It generally run on CPU and is used for post-processing, e.g., NMS, etc. But it can also run on GPU if API server has GPU.

    Builtin module has three methods:
    - initialize: Initialize builtin module. This method is called when API server starts.
    - execute: Execute builtin module. This method is called when API server receives a request.
    - metadata: Return metadata of builtin module. This method is called when API server starts.
    Note that all methods are async.

    All builtin modules are must be implemented in trytune/services/modules/builtin.py.
    When API server starts, all builtin modules will be automatically collected and registered into moudles.
    Note that builtin modules cannot be dynamically loaded.
    """

    # E.g., {"name": "nms", "args": {"threshold": 0.9}}
    @abstractmethod
    async def initialize(self, args: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def execute(self, requests: Any) -> Dict[str, DataSchema]:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError


# Class to store module metadatas and links to triton servers.
class Modules:
    def __init__(self) -> None:
        self.modules: Dict[str, Dict[str, Any]] = {}

    def set(self, module: str, metadata: Dict[str, Any]) -> None:
        assert module not in self.modules
        self.modules[module] = metadata

    def get(self, module: str) -> Dict[str, Any]:
        return self.modules[module]

    # Return all builtin modules can be used in API server.
    # But not yet initialized.
    def get_builtins(self) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError


# FIXME: Avoid using singleton pattern and class variables.
# FIXME: This version of class is not thread-safe.
modules = Modules()
