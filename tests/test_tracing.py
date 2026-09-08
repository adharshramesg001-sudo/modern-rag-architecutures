from rag_compare.tracing import Tracer, TracingConfig


def test_tracer_is_a_safe_noop_without_config():
    tracer = Tracer(None, name="test run", query="hello")
    assert tracer.enabled is False

    span = tracer.step("some_step", as_type="retriever", input="hello")
    span.update(output=[{"a": 1}])
    nested = span.start_observation(name="nested", as_type="tool", input="x")
    nested.update(output="y")
    nested.end()
    span.end()

    trace_url = tracer.finish(output="the answer")
    assert trace_url is None


def test_tracer_degrades_gracefully_with_unreachable_host():
    config = TracingConfig(public_key="pk-test", secret_key="sk-test", host="http://127.0.0.1:9")
    tracer = Tracer(config, name="test run", query="hello")

    # Construction should not raise even though the host can never be reached.
    span = tracer.step("vector_search", as_type="retriever", input="hello")
    span.update(output=[{"title": "doc", "score": 0.5}])
    span.end()

    trace_url = tracer.finish(output="the answer")
    assert trace_url is None  # network unreachable, but no exception was raised
