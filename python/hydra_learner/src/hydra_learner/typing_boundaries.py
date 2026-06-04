from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

type JaxArray = Any
type JaxModule = Any
type MahjaxEnv = Any
type MahjaxStepFn = Any
type TorchCheckpointPayload = Mapping[str, object]
type JsonObject = Mapping[str, object]
type PyO3Module = Any


class MahjaxRoundState(Protocol):
    def __getattr__(self, name: str) -> JaxArray: ...


class MahjaxPlayers(Protocol):
    def __getattr__(self, name: str) -> JaxArray: ...


class MahjaxState(Protocol):
    def __getattr__(self, name: str) -> JaxArray: ...
