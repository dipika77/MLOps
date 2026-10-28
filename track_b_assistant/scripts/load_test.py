"""
Small async load test for the /chat endpoint.

    python scripts/load_test.py --requests 20 --concurrency 5
    python scripts/load_test.py --requests 20 --concurrency 5 --no-cache

It prints latency percentiles, throughput and how many requests were rate limited.
"""
import argparse
import asyncio
import random
import statistics
import time

import httpx

QUESTIONS = [
    "How many paid leaves do I get per year?",
    "What is the price of the Starter plan?",
    "How do I reset my password?",
    "Can I carry forward unused leave?",
    "What is the uptime SLA for Enterprise?",
    "How long are deleted files kept on the Business plan?",
    "What is the meal allowance during travel?",
    "When are performance reviews done?",
]


async def send_request(client, url, use_cache, results):
    payload = {"question": random.choice(QUESTIONS), "use_cache": use_cache}
    start = time.perf_counter()
    try:
        response = await client.post(f"{url}/chat", json=payload)
        elapsed = (time.perf_counter() - start) * 1000
        if response.status_code == 200:
            data = response.json()
            results["ok"].append(elapsed)
            if data.get("cached"):
                results["cached"] += 1
            if data.get("degraded"):
                results["degraded"] += 1
        elif response.status_code == 429:
            results["rate_limited"] += 1
        else:
            results["errors"] += 1
    except Exception:
        results["errors"] += 1


async def main(url, total, concurrency, use_cache):
    results = {"ok": [], "cached": 0, "degraded": 0, "rate_limited": 0, "errors": 0}
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=120) as client:
        async def worker():
            async with semaphore:
                await send_request(client, url, use_cache, results)

        start = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(total)])
        duration = time.perf_counter() - start

    ok = sorted(results["ok"])
    print(f"Requests: {total}, concurrency: {concurrency}, cache: {use_cache}")
    print(f"Success: {len(ok)} | cached: {results['cached']} | degraded: {results['degraded']} | "
          f"rate limited (429): {results['rate_limited']} | errors: {results['errors']}")
    if ok:
        p95_index = min(len(ok) - 1, int(len(ok) * 0.95))
        print(f"Latency ms -> p50: {statistics.median(ok):.0f}, p95: {ok[p95_index]:.0f}, max: {ok[-1]:.0f}")
    print(f"Total time: {duration:.2f}s, throughput: {total / duration:.2f} req/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.url, args.requests, args.concurrency, not args.no_cache))
