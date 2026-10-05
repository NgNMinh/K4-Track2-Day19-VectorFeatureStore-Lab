# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB3 — FastAPI `/search` Endpoint + Latency Benchmark
#
# **Stack:** FastAPI + uvicorn + httpx (client). Searcher từ `app/search.py`.
# Maps to slide §7 (Production Patterns) + deliverable bullets 1, 4.
#
# > Mục tiêu: bọc `Searcher` thành REST API, đo P50/P95/P99 latency, đảm bảo
# > P99 < 50 ms cho hybrid mode (rubric threshold).

# %%
import _setup  # noqa: F401
import statistics
import subprocess
import time
from pathlib import Path

import httpx

# %% [markdown]
# ## 1. Khởi động API server (background)
#
# Trong production thực tế, bạn sẽ chạy `make api` ở terminal riêng. Notebook
# này khởi động uvicorn ở background subprocess và đợi `/healthz` trả ready.

# %%
ROOT = Path(_setup.__file__).resolve().parent.parent
URL = "http://localhost:8000"
proc = None

# Reuse a ready server on :8000; only spawn uvicorn when no API is available.
try:
    r = httpx.get(f"{URL}/healthz", timeout=2.0)
    ready = r.status_code == 200 and r.json().get("ready")
except httpx.HTTPError:
    ready = False

if not ready:
    proc = subprocess.Popen(
        ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"],
        cwd=str(ROOT),
    )
    for _ in range(60):
        try:
            r = httpx.get(f"{URL}/healthz", timeout=2.0)
            ready = r.status_code == 200 and r.json().get("ready")
            if ready:
                break
        except httpx.HTTPError:
            pass
        time.sleep(1)
    if not ready:
        raise RuntimeError("API didn't become ready within 60s")

health = httpx.get(f"{URL}/healthz", timeout=5.0).json()
assert health == {"ready": True, "n_docs": 1000}, health
print("healthz:", health)

# %% [markdown]
# ## 2. Single query — kiểm tra response shape

# %%
sample_query = "cloud computing tự động mở rộng"
r = httpx.get(f"{URL}/search", params={"q": sample_query, "mode": "hybrid"}, timeout=30.0)
r.raise_for_status()
body = r.json()
assert {"mode", "query", "latency_ms", "top_k", "hits"} <= body.keys(), body.keys()
assert body["mode"] == "hybrid" and body["query"] == sample_query
assert all({"doc_id", "title", "text", "score"} <= hit.keys() for hit in body["hits"])
print("response fields:", {key: body[key] for key in ("mode", "query", "latency_ms", "top_k")})
print("top-3 hits:")
for h in body["hits"][:3]:
    print(f"  {h['doc_id']:>14}  score={h['score']:.4f}  {h['title']}")

# %% [markdown]
# ## 3. Warmup + latency benchmark (100 queries × 3 modes)
#
# Gửi 10 truy vấn warmup để nạp model/cache, sau đó dùng 50 golden queries ×
# 2 lượt = 100 calls/mode. Đo `body["latency_ms"]` phía server và wall-clock
# phía client; ngưỡng rubric P99 < 50ms áp dụng cho server-side.
#
# Output: bảng P50/P95/P99 cho 3 mode.

# %%
import json

DATA = ROOT / "data"
golden = [json.loads(l) for l in (DATA / "golden_set.jsonl").open(encoding="utf-8")]


def percentile(values: list[float], p: float) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    return sorted(values)[min(int(n * p), n - 1)]


assert len(golden) == 50, f"Expected 50 golden queries, got {len(golden)}"

warmup_queries = [q["query"] for q in golden[:10]]
for q in warmup_queries:
    r = httpx.get(f"{URL}/search", params={"q": q, "mode": "hybrid"}, timeout=30.0)
    r.raise_for_status()
print(f"Warmup complete: {len(warmup_queries)} queries")


def benchmark_mode(mode: str, reps: int = 2) -> dict[str, float]:
    server_latencies: list[float] = []
    wall_latencies: list[float] = []
    for _ in range(reps):
        for q in golden:
            t0 = time.perf_counter()
            r = httpx.get(f"{URL}/search", params={"q": q["query"], "mode": mode}, timeout=30.0)
            wall_latencies.append((time.perf_counter() - t0) * 1000)
            r.raise_for_status()
            server_latencies.append(float(r.json()["latency_ms"]))
    assert len(server_latencies) == len(golden) * reps
    return {
        "p50_server": percentile(server_latencies, 0.50),
        "p95_server": percentile(server_latencies, 0.95),
        "p99_server": percentile(server_latencies, 0.99),
        "p99_wall":   percentile(wall_latencies, 0.99),
    }


print(f"  {'mode':10}  {'P50':>7}  {'P95':>7}  {'P99':>7}  {'P99(wall)':>9}")
results = {}
for mode in ("keyword", "semantic", "hybrid"):
    res = benchmark_mode(mode)
    results[mode] = res
    print(f"  {mode:10}  {res['p50_server']:>5.1f}ms  {res['p95_server']:>5.1f}ms  "
          f"{res['p99_server']:>5.1f}ms  {res['p99_wall']:>7.1f}ms")

# %% [markdown]
# ## 4. Rubric assertion — hybrid P99 server-side < 50ms

# %%
hybrid_p99 = results["hybrid"]["p99_server"]
print(f"Hybrid P99 server-side: {hybrid_p99:.1f}ms")
if hybrid_p99 < 50:
    print(f"PASS — hybrid P99 < 50ms ({hybrid_p99:.1f}ms)")
else:
    print(f"WARN — hybrid P99 >= 50ms ({hybrid_p99:.1f}ms)")
    print("  Possible causes: cold cache, fastembed model not warm yet, or RRF depth=50 is too aggressive")
    print("  Check: re-run benchmark after 10 warm-up queries; or reduce RRF depth")

# %% [markdown]
# ## 5. Cleanup — stop only the API server started by this notebook

# %%
if proc is not None:
    proc.terminate()
    proc.wait(timeout=5)
    print("API server started by notebook stopped")
else:
    print("Reused existing API server; left it running")

# %% [markdown]
# ## Deliverable evidence
#
# 1. Sample `/search` response includes the required fields and top-3 hits.
# 2. Benchmark output reports server-side P50/P95/P99 and client-side P99.
# 3. Final cell reports whether hybrid P99 server-side is below 50ms.
#
# ---
#
# ## Vibe-coding callout
#
# **Delegate freely:** the FastAPI scaffolding (route definition, Pydantic
# response model, lifespan handler). AI generates this perfectly given the
# spec "GET /search?q=str&mode=Literal[...] returning SearchResponse with
# latency_ms field". `app/main.py` is exactly that pattern — review the diff,
# don't write it from scratch.
#
# **Think hard yourself:** *what to measure*. Server-side latency vs wall-clock
# vs client-side. P50 vs P95 vs P99. Cold vs warm. Single user vs concurrent.
# These are *judgement* decisions: nếu rubric chỉ check P99, optimization sẽ
# hướng vào tail latency, không phải mean. Đừng nhờ AI quyết định metric —
# chỉ nhờ implement metric đã chọn.
