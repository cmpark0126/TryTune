from abc import ABC, abstractmethod
from typing import Any, Dict


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
    async def execute(self, requests: Any) -> Any:
        raise NotImplementedError

    # We use subset of the Triton Server Model Config: https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md
    # Metadata should have the following fields:
    # - inputs, outputs: List of tensors. Each tensor has the following fields:
    #   - name: Name of the tensor
    #   - datatype: Datatype of the tensor. Currently, we only support FP32.
    #   - shape: Shape of the tensor. -1 means dynamic shape.
    # - max_batch_size: Maximum batch size. If max_batch_size is 0, it means module does not support batching.
    # Some information will be added automatically, e.g., name, etc.
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError
