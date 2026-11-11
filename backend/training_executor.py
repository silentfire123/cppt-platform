from __future__ import annotations

"""Framework-neutral asynchronous training executor contract."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Mapping


@dataclass(frozen=True)
class TrainingEvent:
    type: str
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionHandle:
    task_id: str
    workdir: str
    config_path: str
    process: Any = None
    prepared: bool = False


class TrainingExecutor(ABC):
    """Common lifecycle used by local and future remote executors."""

    @abstractmethod
    async def prepare(self, task: Mapping[str, Any]) -> ExecutionHandle:
        raise NotImplementedError

    @abstractmethod
    async def start(self, task: Mapping[str, Any]) -> ExecutionHandle:
        raise NotImplementedError

    @abstractmethod
    async def stop(self, task: Mapping[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def status(self, task: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def events(self, task: Mapping[str, Any]) -> AsyncIterator[TrainingEvent]:
        raise NotImplementedError

    @abstractmethod
    async def collect_artifacts(self, task: Mapping[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def cleanup(self, task: Mapping[str, Any]) -> None:
        raise NotImplementedError


__all__ = ["ExecutionHandle", "TrainingEvent", "TrainingExecutor"]
