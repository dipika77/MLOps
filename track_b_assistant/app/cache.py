"""
Simple in-memory cache for question -> answer.

If the same question is asked again (with the same settings) we return the saved
answer instead of calling the LLM. This makes repeated questions ~instant and saves API quota.

Note: the cache lives inside the API process. If we run many API containers,
Redis would be a better choice so all containers share one cache.
"""
import hashlib
import time
from collections import OrderedDict


class TTLCache:
    def __init__(self, max_items=200, ttl_seconds=600):
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self.data = OrderedDict()  # key -> (saved_time, value)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(question, temperature, top_p):
        # "What is the leave policy?" and "what is the leave policy" should hit the same entry
        normalized = " ".join(question.lower().strip().rstrip("?.!").split())
        raw = f"{normalized}|{temperature}|{top_p}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key):
        item = self.data.get(key)
        if item is None:
            self.misses += 1
            return None

        saved_time, value = item
        if time.time() - saved_time > self.ttl_seconds:
            del self.data[key]  # expired
            self.misses += 1
            return None

        self.data.move_to_end(key)  # mark as recently used
        self.hits += 1
        return value

    def set(self, key, value):
        self.data[key] = (time.time(), value)
        self.data.move_to_end(key)
        # remove the least recently used item when the cache is full
        while len(self.data) > self.max_items:
            self.data.popitem(last=False)

    def clear(self):
        self.data.clear()

    def stats(self):
        return {"items": len(self.data), "hits": self.hits, "misses": self.misses}
