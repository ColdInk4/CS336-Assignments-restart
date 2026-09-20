# utils/results.py

from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    size: str
    mode: str
    status: str = "ok"
    mean_ms: float = float("nan")
    std_ms: float = float("nan")
    n: int = 0
    error: str = ""

    @classmethod
    def failed(cls, size, mode, status, exc):
        msg = str(exc).splitlines()[0][:120] if str(exc) else ""
        return cls(size=size, mode=mode, status=status, error=msg)
