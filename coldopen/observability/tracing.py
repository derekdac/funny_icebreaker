"""OpenTelemetry distributed tracing setup for ColdOpen agents and tools."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


_INITIALIZED = False


def init_tracing(service_name: str = "coldopen", export_to_console: bool = False) -> trace.Tracer:
    """Initialize OpenTelemetry TracerProvider once and return the ColdOpen tracer."""
    global _INITIALIZED
    if not _INITIALIZED:
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        if export_to_console:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        _INITIALIZED = True
    return trace.get_tracer(service_name)


@contextmanager
def traced_span(name: str, attributes: dict[str, str | int | bool] | None = None) -> Iterator[trace.Span]:
    """Context manager creating a linked OpenTelemetry span with safe attributes."""
    tracer = init_tracing()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield span
