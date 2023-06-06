from abc import ABC, abstractmethod
from typing import Any, Dict
from trytune.schemas.common import DataSchema


class StatelessModule(ABC):
    """
    Stateless module interface. Which means the module does not have any state.
    For example, NMS module's result is only determined by the input and threshold.
    """

    # E.g., {"name": "nms", "args": {"threshold": 0.9}}
    @abstractmethod
    def initialize(self, args: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def execute(self, requests: Any) -> Dict[str, DataSchema]:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError


class NmsModule(StatelessModule):
    def initialize(self, args: Dict[str, Any]) -> None:
        self.threshold = args["threshold"]

    def execute(self, requests: Any) -> Dict[str, DataSchema]:
        # If request has threshold, use it, otherwise use self.threshold
        raise NotImplementedError

    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError


class CropModule(StatelessModule):
    def initialize(self, args: Dict[str, Any]) -> None:
        pass

    def execute(self, requests: Any) -> Dict[str, DataSchema]:
        raise NotImplementedError

    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError
