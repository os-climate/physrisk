import importlib.util
import sys
from threading import Lock
from types import ModuleType
from typing import Callable, Generic

from typing_extensions import TypeVar

T = TypeVar("T")


class Lazy(Generic[T]):
    def __init__(self, provider: Callable[[], T]) -> None:
        self._value = None
        self._provider = provider
        self._lock = Lock()

    def set_provider(self, provider: Callable[[], T]):
        """Update provider.

        Args:
            provider (Callable[..., T]): Provider to use.
        """
        with self._lock:
            self._provider = provider

    def value(self):
        """Get value, loading as needed.

        Returns:
            T: Value.
        """
        with self._lock:
            if self._value is None:
                self._value = self._provider()
        return self._value


def lazy_import(name):
    spec = importlib.util.find_spec(name)
    if spec is not None:
        spec_loader = spec.loader
        assert spec_loader is not None
        loader = importlib.util.LazyLoader(spec_loader)
        spec.loader = loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        loader.exec_module(module)
        return module
    return ModuleType("not found", None)
