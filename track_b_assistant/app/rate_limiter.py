"""
Sliding window rate limiter.

Each client (IP address) can make at most `max_requests` requests in `window_seconds`.
We keep the timestamps of recent requests for every client and drop the old ones.
"""
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, max_requests=20, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)

    def is_allowed(self, client_id):
        now = time.time()
        timestamps = self.requests[client_id]

        # remove requests that are outside the time window
        while timestamps and now - timestamps[0] > self.window_seconds:
            timestamps.popleft()

        if len(timestamps) >= self.max_requests:
            return False

        timestamps.append(now)
        return True

    def retry_after(self, client_id):
        """Seconds until the client can send the next request."""
        timestamps = self.requests[client_id]
        if not timestamps:
            return 0
        return max(0, int(self.window_seconds - (time.time() - timestamps[0])) + 1)
