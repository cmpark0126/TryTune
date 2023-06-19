from typing import Any, Dict


# Class to store pipeline metadatas and links to triton servers.
class Pipelines:
    def __init__(self) -> None:
        # All pipelines installed in the system.
        self.pipelines: Dict[str, Dict[str, Any]] = {}

    def set(self, pipeline: str, metadata: Dict[str, Any]) -> None:
        assert pipeline not in self.pipelines
        self.pipelines[pipeline] = metadata

    def get(self, pipeline: str) -> Dict[str, Any]:
        return self.pipelines[pipeline]


# FIXME: Avoid using singleton pattern and class variables.
# FIXME: This version of class is not thread-safe.
pipelines = Pipelines()
