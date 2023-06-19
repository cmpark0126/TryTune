import inspect
from typing import Any, Dict

from trytune.services.moduels import builtins, common


# Class to store module metadatas and links to triton servers.
class Modules:
    def __init__(self) -> None:
        # All modules installed in the system.
        self.modules: Dict[str, Dict[str, Any]] = {}
        # All builtin modules available in the system. But, not yet installed.
        self.available_builtins: Dict[str, Any] = {}

        for name, obj in inspect.getmembers(builtins):
            if (
                inspect.isclass(obj)
                and issubclass(obj, common.BuiltinModule)
                and name != "BuiltinModule"
            ):
                metadata = obj().metadata()
                metadata["name"] = name
                metadata["is_builtin"] = True
                print(f"Found builtin module: {name}: {metadata}")

                self.available_builtins[name] = {"metadata": metadata, "object": obj}
                # FIXME: convert to debug log

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
