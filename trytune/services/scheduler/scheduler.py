from abc import ABC, abstractmethod


class SchedulerInner(ABC):
    @abstractmethod
    async def send_inference_request(self, model: str, request: dict) -> dict:
        pass
