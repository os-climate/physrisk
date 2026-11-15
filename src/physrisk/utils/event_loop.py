import asyncio
import concurrent.futures
import os
import threading
from contextlib import contextmanager
from typing import Any, Coroutine, Optional, TypeVar

from physrisk.utils.concurrency import available_cpu_count

T = TypeVar("T")


# from https://github.com/fsspec/filesystem_spec/blob/master/fsspec/asyn.py
@contextmanager
def _selector_policy():
    original_policy = asyncio.get_event_loop_policy()
    try:
        if os.name == "nt" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        yield
    finally:
        asyncio.set_event_loop_policy(original_policy)


def _running_loop() -> Optional[asyncio.AbstractEventLoop]:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to completion from synchronous code.

    If the caller is already inside a running event loop (e.g. a synchronous
    method called from async code), the coroutine runs on a new loop in a
    worker thread, since the running loop cannot be blocked on.

    Args:
        coro: Coroutine to run.

    Returns:
        Result of the coroutine.
    """
    if _running_loop() is None:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


class BackgroundEventLoop:
    """An event loop on its own daemon thread, shared by all callers in the process.

    Work submitted from any thread or event loop runs on this loop, so resources
    bound to a loop (e.g. an async S3 client) and limits on concurrency are shared.
    The loop's default executor (used by asyncio.to_thread on this loop) is sized
    separately from any caller's. The loop starts on first use, and again in a
    forked child, which does not inherit the parent's thread.
    """

    _instances: "list[BackgroundEventLoop]" = []

    def __init__(self, name: str, executor_workers: Optional[int] = None):
        """
        Args:
            name: Name of the loop's thread.
            executor_workers: Size of the loop's default executor. Defaults to the
                CPUs available to the process.
        """
        self._name = name
        self._executor_workers = executor_workers
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()
        BackgroundEventLoop._instances.append(self)

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            with self._lock:
                if self._loop is None:
                    with _selector_policy():
                        loop = asyncio.new_event_loop()
                    loop.set_default_executor(self._new_executor())
                    threading.Thread(
                        target=loop.run_forever, name=self._name, daemon=True
                    ).start()
                    self._loop = loop
        return self._loop

    def is_current(self) -> bool:
        """True if called from a coroutine running on this loop."""
        return self._loop is not None and _running_loop() is self._loop

    def set_executor_workers(self, workers: Optional[int]) -> None:
        """Resize the loop's default executor; tasks already submitted are unaffected."""
        with self._lock:
            self._executor_workers = workers
            if self._loop is not None:
                self._loop.call_soon_threadsafe(
                    self._loop.set_default_executor, self._new_executor()
                )

    def _new_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        return concurrent.futures.ThreadPoolExecutor(
            max_workers=self._executor_workers or available_cpu_count(),
            thread_name_prefix=self._name,
        )

    async def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Await a coroutine on this loop from any other event loop.

        Cancelling the caller's await cancels the coroutine on this loop.
        """
        loop = self.loop
        if _running_loop() is loop:
            return await coro
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, loop))

    def run_sync(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run a coroutine on this loop, blocking until it completes.

        Must not be called from this loop's own thread.
        """
        loop = self.loop
        if _running_loop() is loop:
            coro.close()
            raise RuntimeError(f"run_sync called on the {self._name} loop")
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    @classmethod
    def _reset_after_fork(cls) -> None:
        for instance in cls._instances:
            instance._loop = None
            instance._lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=BackgroundEventLoop._reset_after_fork)
