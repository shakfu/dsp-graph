# Threading Plan: Offload Blocking Work Off the Async Event Loop

Status: proposed
Scope: `src/dsp_graph/api/build.py`, `api/simulate.py`, `api/optimize.py`, `cache.py`
Related TODO: "Code review follow-ups > Robustness & ops" (items 1 and 2)

## Problem

Every request handler is declared `async def` (28 of them), yet several perform
long, blocking, CPU- or IO-bound work directly in the coroutine body. There is
zero use of `run_in_threadpool` / `anyio.to_thread` anywhere in the codebase.

Because FastAPI runs `async def` handlers directly on the single asyncio event
loop, any blocking call inside them stalls the *entire* server until it returns.
The offenders, worst first:

| Handler | Blocking work | Typical cost |
|---|---|---|
| `POST /build`, `/build/binary` | `_compile_build_cached` -> `ProjectGenerator.generate` + native `Builder(...).build()` + disk cache read/write | seconds to tens of seconds |
| `POST /build/batch` | loop of `_compile_build_cached` over N platforms | N x the above |
| `GET /build/batch/{id}/zip` | `BuildCache.get` disk reads + in-memory zip of all artifacts | up to hundreds of MB |
| `POST /simulate`, `/simulate/continue` | numpy `run_sim` over up to 262 144 samples | tens to hundreds of ms |
| `POST /optimize`, `/optimize/pass` | `optimize_graph` / single pass (pure-Python graph walks) | ms to tens of ms |

The user-visible consequence: the editor debounces a parse+validate on every
keystroke. While any build runs, those `/api/graph/load/gdsp` and
`/api/graph/validate` requests queue behind it, so the editor appears frozen for
the entire build. This is a correctness/availability defect in the interactive
core, not a feature gap.

## Key insight from investigation (why the design is narrower than it first looks)

My first instinct was "wrap the handlers in threads and put a `Lock` around the
three module-global mutable stores (`_sessions`, `_batch_cache`, the
`BuildCache` singleton)." Reading the code changed that conclusion:

- Under CPython, the GIL plus the single event loop means **dict operations that
  stay on the loop are already serialized** and cannot corrupt each other. They
  only yield at `await` points.
- The *only* true parallelism this change introduces is whatever runs **inside**
  the threadpool. So thread-safety analysis reduces to: "what shared mutable
  state do the offloaded functions touch?"
- Therefore the correct design is to **offload only the blocking compute/IO core
  and keep all module-global dict access on the event loop.** That eliminates the
  need to lock `_sessions` and `_batch_cache` entirely.

What the offloaded functions actually touch:

- `run_sim` (numpy): touches only the per-call `SimState`/`Graph` locals. No
  shared in-memory state. Safe to run in parallel.
- `_compile_build_cached`: touches the `BuildCache` singleton's **disk** state
  via `get`/`put`/`_evict_expired`. `_cache_instance` itself is assigned once and
  thereafter read-only. So the real concurrency question is confined to
  `BuildCache`'s filesystem operations.

This narrows the locking scope from "three stores" to "one: `BuildCache`'s
mutating disk operations."

## Design

### 1. Offload the blocking core, keep dict access on the loop

Pattern for each offender: keep request parsing, validation, `HTTPException`
raising, and all `_sessions` / `_batch_cache` mutation in the `async def` body on
the loop. Move only the heavy pure function into a thread:

```python
from fastapi.concurrency import run_in_threadpool

@router.post("/build", response_model=CompileBuildResponse)
async def compile_build(req: GenerateRequest) -> CompileBuildResponse:
    g = _validate_generate_request(req)          # on loop
    async with _build_slot:                       # bounded-concurrency gate (below)
        try:
            cr = await run_in_threadpool(_compile_build_cached, g, req.platform)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return cr.response
```

Handlers to convert (offload only the marked call):

- `build.py::compile_build` / `download_binary` -> offload `_compile_build_cached`
- `build.py::batch_build` -> offload the per-platform `_compile_build_cached`
  calls (see note under bounded concurrency); keep the `_batch_cache` write on
  the loop
- `build.py::download_batch_zip` -> offload the "read artifacts + build zip"
  loop (a small helper `_zip_batch_artifacts(entry) -> bytes`); keep the
  `_batch_cache` `.get`/`del` on the loop
- `simulate.py::simulate` / `simulate_continue` -> offload `run_sim`; keep
  `_sessions` read/write and `_cleanup_sessions` on the loop
- `optimize.py::optimize` / `optimize_pass` -> offload `optimize_graph` /
  `PASSES[name]` (lower priority; ms-scale, but cheap to include for consistency)

`buffer_get/set`, `simulate_param`, `simulate_peek`, `simulate_reset` operate on
already-resident state in microseconds; leave them on the loop.

Rationale for `run_in_threadpool` over converting to plain `def` handlers:
FastAPI would also threadpool a `def` handler, but that would push the
`_sessions` / `_batch_cache` mutations into the thread too, re-introducing the
locking problem we just designed away. Explicit `run_in_threadpool` around the
pure core keeps shared-state access on the loop. This is the deciding factor.

### 2. Bound native-build concurrency

Native builds are heavy (full toolchain per build). Starlette's default anyio
threadpool allows 40 concurrent threads; letting 40 native compilers run at once
would thrash CPU/IO and could OOM. Add a small asyncio semaphore to cap
concurrent builds:

```python
import asyncio
_MAX_CONCURRENT_BUILDS = 2            # tune; or os.cpu_count()-based
_build_slot = asyncio.Semaphore(_MAX_CONCURRENT_BUILDS)
```

Acquire it in `compile_build`, `download_binary`, and around **each** platform
build inside `batch_build` (so a batch does not monopolize all slots and starve
single builds — acquire/release per platform, not once for the whole batch).
Simulation and optimize do not need this gate.

Note: an `asyncio.Semaphore` is awaited on the loop *before* entering the
thread, so it is not itself a threading primitive and needs no lock.

### 3. Make `BuildCache` mutating disk ops thread-safe

`get`, `put`, `_evict_expired`, `clear`, `size` can now run on multiple
threadpool threads simultaneously (distinct or identical cache keys). Current
state:

- `put` already uses write-to-temp + atomic `rename`, and tolerates the target
  appearing mid-rename. Good.
- The real hazard is `_evict_expired` / `clear` calling `shutil.rmtree(shard)`
  on a shard that another thread is mid-`get` (reading `meta.json` / artifact),
  causing a spurious miss or an `OSError`. `get` already swallows `OSError` and
  returns `None`, so the blast radius is "occasional false cache miss ->
  redundant rebuild," not corruption.

Two options:

- (a) Add a single `threading.Lock` around the mutating sections of `put`,
  `_evict_expired`, `clear`. Simple; serializes cache bookkeeping but not the
  builds themselves (builds happen outside the cache calls). Recommended default.
- (b) Per-key build dedup lock: a `dict[str, threading.Lock]` guarded by a meta
  `threading.Lock`, so two identical concurrent builds share one lock and the
  second gets the cache hit instead of rebuilding (thundering-herd fix). More
  code; do only if identical-concurrent-builds is observed to matter.

Recommend (a) now; note (b) as a follow-up. Add the lock as a `BuildCache`
instance attribute (`self._lock = threading.Lock()`) so per-instance test caches
stay isolated.

### 4. Residual logical race (document, do not over-engineer)

Two concurrent requests for the **same** `session_id` can both read the old
`SimState` on the loop, both run `run_sim` in parallel threads, and the later
write wins — advancing the session by only one call's worth of samples. This
race exists today (the `await` boundary just moves) and matters only if a client
fires overlapping requests on one session, which the frontend does not. Document
it in a comment; do not add per-session locking unless it becomes real.

## Test plan

Add `tests/test_threading.py`:

1. **Event loop not blocked during a build** (the core regression test). Monkey-
   patch `_compile_build_cached` with a version that `time.sleep`s ~0.5 s. Using
   an async client (`httpx.ASGITransport` + `asyncio.gather`), fire one `/build`
   concurrently with several `/api/graph/validate` calls; assert the validate
   responses return well before the build completes (wall-clock check with a
   generous margin). Fails on `main` (serialized), passes after offload.
2. **Concurrent identical + distinct builds** don't corrupt the cache. Drive
   `BuildCache.put`/`get` from a `ThreadPoolExecutor` with overlapping and
   distinct keys interleaved with `_evict_expired`; assert every present key
   reads back correct bytes and no unexpected exception escapes.
3. **Build concurrency is bounded.** Patch the build core to record concurrent
   entry count via a shared counter; fire more concurrent `/build` requests than
   `_MAX_CONCURRENT_BUILDS`; assert observed max concurrency <= the cap.
4. **Session mutation stays correct** under concurrent distinct-session
   `/simulate` calls: fire K sessions in parallel, assert K distinct
   `session_id`s and correct per-session output, and that `_MAX_SESSIONS`
   eviction still holds.

Keep existing `tests/test_api_build.py` and `test_api_simulate.py` green
unchanged — the offload must be behavior-preserving for the sequential path.

## Rollout / ordering

1. `BuildCache` lock (item 3) — smallest, isolated, independently testable.
2. Offload `simulate` + `optimize` (item 1, no new shared state) + test 1/4.
3. Offload `build`/`batch`/`zip` + bounded semaphore (items 1+2) + tests 2/3.
4. `make qa` (test + lint + typecheck + format) green.

## Out of scope (tracked separately in TODO)

- Multi-worker deployment: this plan keeps the single-worker, in-process model.
  `_sessions`, `_batch_cache`, and the cache singleton remain per-process. The
  "document the single-worker assumption" TODO item still stands; a shared store
  (Redis/DB) is a separate, larger change.
- Gating `/api/build*` behind `--enable-build` (separate TODO item).

## Risks / trade-offs

- `run_in_threadpool` adds a small per-call thread-hop latency (sub-ms) to the
  offloaded endpoints. Negligible against seconds-long builds; for optimize
  (ms-scale) it is borderline — acceptable for consistency, or leave optimize on
  the loop if the hop is measurable. Recommend offloading it but flag as the one
  discretionary call.
- Bounding builds to 2 concurrent means the 3rd concurrent build *waits* (it does
  not fail). That is the intended back-pressure; document it so it is not read as
  a hang.
- The `BuildCache` lock serializes cache bookkeeping across threads. Bookkeeping
  is fast (stat/rename/rmtree of small dirs); the expensive build runs outside
  the lock, so throughput is unaffected.
