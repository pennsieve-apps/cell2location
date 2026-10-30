"""
Container entry point: fixes the thread count from THREADS (or the cgroup CPU quota, never the
host's core count) before numpy or torch is imported, then runs main() once.
"""
import os
import sys

from processor.resources import resolve_threads, set_thread_env

_raw = os.environ.get("THREADS", "").strip()
_n, _source = resolve_threads(int(_raw) if _raw.isdigit() else 0)
set_thread_env(_n)

from processor.main import main  # noqa: E402

sys.exit(main())
