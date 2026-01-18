import asyncio
import os
import threading
from contextlib import contextmanager
from typing import Optional

iothread = [None]  # dedicated fsspec IO thread
loop = [None]  # global event loop for any non-async instance
_lock = None  # global lock placeholder
get_running_loop = asyncio.get_running_loop

# Note that this async pattern is taken from https://github.com/fsspec/filesystem_spec/blob/master/fsspec/asyn.py
#


def get_lock():
    """Allocate or return a threading lock.

    The lock is allocated on first use to allow setting one lock per forked process.
    """
    global _lock
    if not _lock:
        _lock = threading.Lock()
    return _lock


def reset_lock():
    """Reset the global lock.

    This should be called only on the init of a forked process to reset the lock to
    None, enabling the new forked process to get a new lock.
    """
    global _lock

    iothread[0] = None
    loop[0] = None
    _lock = None


@contextmanager
def _selector_policy():
    original_policy = asyncio.get_event_loop_policy()
    try:
        if os.name == "nt" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        yield
    finally:
        asyncio.set_event_loop_policy(original_policy)


def get_loop():
    """Create or return the default API IO loop.

    The loop is run on a separate thread.
    """
    if loop[0] is None:
        with get_lock():
            # repeat the check just in case the loop got filled between the
            # previous two calls from another thread
            if loop[0] is None:
                with _selector_policy():
                    loop[0] = asyncio.new_event_loop()
                th = threading.Thread(
                    target=loop[0].run_forever, name="esg-physrisk-IO"
                )
                th.daemon = True
                th.start()
                iothread[0] = th
    return loop[0]


async def _runner(event: threading.Event, coro, result, timeout=None):
    timeout = timeout if timeout else None  # convert 0 or 0.0 to None
    if timeout is not None:
        coro = asyncio.wait_for(coro, timeout=timeout)
    try:
        result[0] = await coro
    except Exception as ex:
        result[0] = ex
    finally:
        event.set()


def run(coro, loop, timeout: Optional[float] = None):
    """Run a coroutine in the given event loop, waiting until this is complete or has timed out.
    This is blocking, typically used within a thread pool to execute a batch of IO requests executed in
    parallel.

    Args:
        coro: Coroutine.
        loop: Event loop.
        timeout (Optional[float], optional): Timeout. Defaults to None.

    Raises:
        asyncio.TimeoutError: Raised on timeout.

    Returns:
        Result of running coroutine.
    """
    result = [None]
    event = threading.Event()
    asyncio.run_coroutine_threadsafe(_runner(event, coro, result, timeout), loop)
    while True:
        # this loops allows thread to get interrupted
        if event.wait(1):
            break
        if timeout is not None:
            timeout -= 1
            if timeout < 0:
                raise asyncio.TimeoutError()

    return result[0]
