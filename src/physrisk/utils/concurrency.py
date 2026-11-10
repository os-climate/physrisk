import math
import os
from pathlib import Path
from typing import Optional


def available_cpu_count() -> int:
    """Return the number of CPUs this process may actually use.

    Unlike ``os.cpu_count()``, this respects CPU affinity and container CPU quotas
    (cgroup v2 ``cpu.max`` or cgroup v1 ``cpu.cfs_quota_us``). In a container
    limited to one CPU on a large host, ``os.cpu_count()`` reports the host's CPUs,
    and thread pools sized from it oversubscribe the single CPU.

    Returns:
        Number of usable CPUs, at least 1.
    """
    try:
        count = len(os.sched_getaffinity(0))
    except AttributeError:  # not available on macOS or Windows
        count = os.cpu_count() or 1
    quota = _cgroup_cpu_quota()
    if quota is not None:
        count = min(count, max(1, math.ceil(quota)))
    return max(1, count)


def _cgroup_cpu_quota() -> Optional[float]:
    """CPU quota of the process's cgroup, or None if unlimited or unknown."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:
        quota_us = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period_us = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        return None if quota_us <= 0 else quota_us / period_us
    except (OSError, ValueError):
        return None
