"""Fixed-rate loop limiter for deployment control loops."""

from __future__ import annotations

import time


class RateLimiter:
    def __init__(self, frequency_hz: float) -> None:
        if frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be greater than zero.")
        self._period_s = 1.0 / frequency_hz
        self._next_time = time.perf_counter()

    def sleep(self) -> None:
        now = time.perf_counter()
        if self._next_time <= now:
            self._next_time = now + self._period_s
            return
        time.sleep(self._next_time - now)
        self._next_time += self._period_s
