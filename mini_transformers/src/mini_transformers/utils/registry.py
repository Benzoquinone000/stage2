"""Generic registry."""

from __future__ import annotations

from typing import Callable, Generic, TypeVar


T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self) -> None:
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(item: T) -> T:
            self._items[name] = item
            return item

        return decorator

    def get(self, name: str) -> T:
        return self._items[name]

    def keys(self) -> list[str]:
        return sorted(self._items)
