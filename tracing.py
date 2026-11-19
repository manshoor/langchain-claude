"""Langfuse tracing for the support workflow.

Off unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are both set.
LANGFUSE_BASE_URL picks the region (default EU cloud) or a self-hosted server.
This is the only module that imports langfuse, and only when tracing is on.

    turn_trace() ──► top-level "agent" observation, trace ID fixed up front
       │              session = thread, tags = workflow, preset:<name>
       └─ trace.callbacks ──► CallbackHandler ──► every graph step and model call, nested

run_experiment() runs an eval over many items: a Langfuse experiment run (one
trace per item, compared side by side in the UI) with keys, a local thread
pool without.

A Langfuse failure is logged and ignored: it never fails a customer turn.
The Anthropic SDK's own OpenTelemetry spans are dropped: LangChain already
records each model call, so keeping both would count every call twice.
Data is sent in the background; shutdown() flushes it before the script exits.
"""

import asyncio
import functools
import logging
import os
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from typing import Any

KEY_VARS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
CATEGORICAL = "CATEGORICAL"
NUMERIC = "NUMERIC"
ANTHROPIC_SDK_SCOPE = "com.anthropic.sdk.python"

_log = logging.getLogger(__name__)

# An eval task: (item, callbacks to pass to LangChain) -> output.
Task = Callable[[dict, list], Any]


def enabled() -> bool:
    """True when both Langfuse keys are set."""
    return all(os.environ.get(name) for name in KEY_VARS)


class Trace:
    """One customer turn. Every method does nothing when tracing is off or failed."""

    def __init__(self, trace_id: str | None = None, span: Any = None, handler: Any = None):
        self._id = trace_id
        self._span = span
        self._handler = handler

    @property
    def callbacks(self) -> list:
        """Pass these in the graph's config to trace every step."""
        return [self._handler] if self._handler else []

    def score(self, name: str, value: float | str) -> None:
        """Attach a score by trace ID (safe with parallel steps, unlike last_trace_id)."""
        if self._id is None:
            return

        _safely(_create_score, self._id, name, value)

    def finish(self, output: str) -> None:
        """Record the reply the customer saw."""
        if self._span is None:
            return

        _safely(self._span.update, output=output)

    def url(self) -> str | None:
        """Link to this trace in the Langfuse UI."""
        if self._id is None:
            return None

        return _safely(_trace_url, self._id)


_OFF = Trace()


@contextmanager
def turn_trace(name: str, session_id: str, tags: list[str], text: str) -> Iterator[Trace]:
    """Trace everything run inside this block as one turn."""
    if not enabled():
        yield _OFF
        return

    # Setup errors leave tracing off for this turn; the turn itself still runs.
    stack = ExitStack()
    trace = _safely(_start, stack, name, session_id, tags, text) or _OFF
    try:
        yield trace
    finally:
        _safely(stack.close)


def shutdown() -> None:
    """Send any queued traces. Call once before the script exits."""
    if not enabled():
        return

    _safely(_shutdown)


def run_experiment(
    name: str, run_name: str, items: list[dict], task: Task, workers: int
) -> list[tuple[Any, str | None]]:
    """Run `task` on every item. Returns (output, trace ID) per item, in item order.

    A failed item gives output None; the others still run. Each item needs
    an "input" key (Langfuse's experiment format).
    """
    # None = not run yet. Langfuse fills in items as they finish.
    results: list[tuple[Any, str | None] | None] = [None] * len(items)
    if enabled():
        _safely(_langfuse_experiment, name, run_name, items, task, workers, results)

    # Tracing off, or Langfuse failed part way: run only what is missing, so
    # no paid item runs twice.
    missing = [i for i, result in enumerate(results) if result is None]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outputs = pool.map(lambda i: _run_item(task, items[i], []), missing)
        for i, output in zip(missing, outputs, strict=True):
            results[i] = (output, None)

    return results


def _run_item(task: Task, item: dict, callbacks: list) -> Any:
    """Run one eval item; log and return None on any error."""
    try:
        return task(item, callbacks)
    except Exception as error:  # noqa: BLE001 - one bad item must not stop the eval
        _log.warning("Eval item %r failed: %s", item.get("input"), error)
        return None


# ---- Langfuse calls (lazy imports: nothing loads without keys) ---------------


@functools.cache
def _client():
    """The one Langfuse client, created on first use with the span filter."""
    from langfuse import Langfuse

    return Langfuse(should_export_span=_keep_span)


def _keep_span(span) -> bool:
    """Langfuse's default export rule, minus the Anthropic SDK's duplicate spans."""
    from langfuse.span_filter import is_default_export_span

    scope = span.instrumentation_scope
    if scope is not None and scope.name == ANTHROPIC_SDK_SCOPE:
        return False

    return is_default_export_span(span)


def _start(stack: ExitStack, name: str, session_id: str, tags: list[str], text: str) -> Trace:
    """Open the top-level observation and set session and tags for everything under it."""
    from langfuse import propagate_attributes
    from langfuse.langchain import CallbackHandler

    client = _client()
    trace_id = client.create_trace_id()
    span = stack.enter_context(
        client.start_as_current_observation(
            trace_context={"trace_id": trace_id}, name=name, as_type="agent", input=text
        )
    )
    stack.enter_context(propagate_attributes(session_id=session_id, tags=tags, trace_name=name))
    return Trace(trace_id, span, CallbackHandler())


def _langfuse_experiment(
    name: str, run_name: str, items: list[dict], task: Task, workers: int, results: list
) -> None:
    """Run the items as one Langfuse experiment run, `workers` at a time.

    Writes (output, trace ID) into `results` by item index as items finish.
    """
    from langfuse.langchain import CallbackHandler

    # Langfuse awaits the task in its event loop: run ours in a thread so
    # items overlap. to_thread copies the context, so traces still nest.
    async def langfuse_task(*, item: dict, **kwargs: Any) -> Any:
        output = await asyncio.to_thread(_run_item, task, item, [CallbackHandler()])
        results[item["metadata"]["index"]] = (output, None)
        return output

    indexed = [
        item | {"metadata": {**item.get("metadata", {}), "index": i}}
        for i, item in enumerate(items)
    ]
    result = _client().run_experiment(
        name=name, run_name=run_name, data=indexed, task=langfuse_task, max_concurrency=workers
    )

    # Add each item's trace ID, to attach scores to later.
    for item_result in result.item_results:
        results[item_result.item["metadata"]["index"]] = (
            item_result.output,
            item_result.trace_id,
        )


def _create_score(trace_id: str, name: str, value: float | str) -> None:
    data_type = CATEGORICAL if isinstance(value, str) else NUMERIC
    _client().create_score(name=name, value=value, trace_id=trace_id, data_type=data_type)


def _trace_url(trace_id: str) -> str | None:
    return _client().get_trace_url(trace_id=trace_id)


def _shutdown() -> None:
    _client().shutdown()


def _safely(call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a Langfuse call; log and return None on any error."""
    try:
        return call(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 - tracing must never fail a turn
        _log.warning("Langfuse: %s", error)
        return None
