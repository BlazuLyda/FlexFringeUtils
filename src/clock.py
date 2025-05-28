import time

class SimpleClock:

    def __init__(self):
        self._start_time = 0
        self._last_measure_time = 0
        self._end_time = 0
        self._running = False

    def start(self):
        """Start the clock."""
        self._start_time = time.time()
        self._last_measure_time = self._start_time
        self._running = True

    def measure(self):
        """Return time since last measurement in human-readable format."""
        if not self._running:
            raise RuntimeError("Clock has not been started.")
        now = time.time()
        elapsed = now - self._last_measure_time
        self._last_measure_time = now
        return self._format_time(elapsed)

    def stop(self):
        """Stop the clock and return total time since start in human-readable format."""
        if not self._running:
            raise RuntimeError("Clock has not been started.")
        self._end_time = time.time()
        self._running = False
        total_time = self._end_time - self._start_time
        return self._format_time(total_time)

    def _format_time(self, seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes}m {seconds}s"
