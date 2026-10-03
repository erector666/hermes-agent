"""Behavioral tests for the final normal-tool execution interception seam."""

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_cli.plugins import ExecutionDecision


def _context(name="interceptor-test"):
    manager = PluginManager()
    return PluginContext(PluginManifest(name=name), manager), manager


def test_interceptor_sees_final_args_and_can_handle_without_native_dispatch(monkeypatch):
    import model_tools
    ctx, manager = _context()
    seen = []
    ctx.register_tool_execution_interceptor(
        lambda tool_name, args, **ids: (seen.append((tool_name, dict(args), ids)) or ExecutionDecision.handled("sentinel")),
        tool_names={"write_file"},
    )
    import hermes_cli.plugins as plugins
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(model_tools, "coerce_tool_args", lambda name, args: args)
    from hermes_cli import middleware
    from hermes_cli.middleware import TOOL_REQUEST_MIDDLEWARE, TOOL_EXECUTION_MIDDLEWARE
    from hermes_cli.plugins import invoke_middleware
    monkeypatch.setattr(plugins, "invoke_middleware", lambda kind, **kw: [cb(**kw) for cb in manager._middleware.get(kind, [])])
    request_seen = []
    def request_rewrite(args, **kw):
        request_seen.append(dict(args))
        return {"args": dict(args)}
    def execution_rewrite(args, next_call, **kw):
        return next_call({"path": "B"})
    manager._middleware[TOOL_REQUEST_MIDDLEWARE] = [request_rewrite]
    manager._middleware[TOOL_EXECUTION_MIDDLEWARE] = [execution_rewrite]
    native = []
    monkeypatch.setattr(model_tools.registry, "dispatch", lambda *a, **kw: native.append((a, kw)) or (_ for _ in ()).throw(AssertionError("native dispatched")))
    result = model_tools.handle_function_call("write_file", {"path": "A"}, task_id="task", session_id="session", turn_id="turn", tool_call_id="call", api_request_id="request")
    assert request_seen == [{"path": "A"}]
    assert result == "sentinel"
    assert seen == [("write_file", {"path": "B"}, {"task_id": "task", "session_id": "session", "turn_id": "turn", "tool_call_id": "call", "api_request_id": "request"})]
    assert native == []


def _prepare_call(monkeypatch, callbacks=(), *, middleware=()):
    import model_tools
    import hermes_cli.plugins as plugins
    from hermes_cli.middleware import TOOL_EXECUTION_MIDDLEWARE

    ctx, manager = _context()
    for callback in callbacks:
        ctx.register_tool_execution_interceptor(callback, tool_names={"read_file"})
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(model_tools, "coerce_tool_args", lambda name, args: args)
    monkeypatch.setattr(plugins, "invoke_middleware", lambda kind, **kw: [cb(**kw) for cb in manager._middleware.get(kind, [])])
    manager._middleware[TOOL_EXECUTION_MIDDLEWARE] = list(middleware)
    native = []
    monkeypatch.setattr(model_tools.registry, "dispatch", lambda *a, **kw: native.append((a, kw)) or "native-result")
    return model_tools, native


def test_continue_runs_native_read_once(monkeypatch):
    from hermes_cli.plugins import ExecutionDecision
    model_tools, native = _prepare_call(monkeypatch, [lambda *a, **kw: ExecutionDecision.continue_()])
    result = model_tools.handle_function_call("read_file", {"path": "sample.txt"})
    assert result == "native-result"
    assert len(native) == 1
    assert native[0][0] == ("read_file", {"path": "sample.txt"})


def test_block_denies_without_native_dispatch(monkeypatch):
    from hermes_cli.plugins import ExecutionDecision
    model_tools, native = _prepare_call(monkeypatch, [lambda *a, **kw: ExecutionDecision.block("policy denied")])
    result = model_tools.handle_function_call("read_file", {"path": "sample.txt"})
    assert "policy denied" in result
    assert native == []


def test_interceptor_block_preserves_denial_through_result_transform(monkeypatch):
    import model_tools
    import hermes_cli.lifecycle as lifecycle
    from hermes_cli.plugins import ExecutionDecision

    model_tools, native = _prepare_call(
        monkeypatch, [lambda *a, **kw: ExecutionDecision.block("policy denied")]
    )
    observed = []
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name in {"post_tool_call", "transform_tool_result"})

    def invoke_hook(name, **kwargs):
        observed.append((name, kwargs))
        if name == "transform_tool_result":
            return ["success: operation completed"]
        return []

    monkeypatch.setattr(lifecycle, "invoke_hook", invoke_hook)
    result = model_tools.handle_function_call(
        "read_file", {"path": "sample.txt"}, task_id="task", session_id="session",
        turn_id="turn", tool_call_id="call", api_request_id="request",
    )

    assert "policy denied" in result
    assert "success" not in result
    assert native == []
    post = next(kwargs for name, kwargs in observed if name == "post_tool_call")
    assert post["status"] == "blocked"
    assert post["error_type"] == "plugin_block"
    assert post["error_message"] == "policy denied"
    assert not any(name == "transform_tool_result" for name, _ in observed)


def test_handled_returns_supplied_result_without_native_dispatch(monkeypatch):
    from hermes_cli.plugins import ExecutionDecision
    model_tools, native = _prepare_call(monkeypatch, [lambda *a, **kw: ExecutionDecision.handled("adapter-result")])
    assert model_tools.handle_function_call("read_file", {"path": "sample.txt"}) == "adapter-result"
    assert native == []


def test_multiple_matching_interceptors_block_before_callbacks(monkeypatch):
    from hermes_cli.plugins import ExecutionDecision
    calls = []
    callbacks = [lambda *a, **kw: calls.append("first") or ExecutionDecision.continue_(),
                 lambda *a, **kw: calls.append("second") or ExecutionDecision.continue_()]
    model_tools, native = _prepare_call(monkeypatch, callbacks)
    result = model_tools.handle_function_call("read_file", {"path": "sample.txt"})
    assert "Multiple execution interceptors" in result
    assert calls == []
    assert native == []


@__import__("pytest").mark.parametrize("decision", [
    None,
    type("UnknownDecision", (), {"action": "UNKNOWN", "reason": None, "result": None})(),
    # CONTINUE must not carry a result; BLOCK must have a non-empty reason.
    __import__("hermes_cli.plugins", fromlist=["ExecutionDecision"]).ExecutionDecision("CONTINUE", result="unexpected"),
    __import__("hermes_cli.plugins", fromlist=["ExecutionDecision"]).ExecutionDecision("BLOCK", reason=""),
    __import__("hermes_cli.plugins", fromlist=["ExecutionDecision"]).ExecutionDecision("HANDLED", result=17),
])
def test_malformed_decisions_fail_closed(monkeypatch, decision):
    model_tools, native = _prepare_call(monkeypatch, [lambda *a, **kw: decision])
    result = model_tools.handle_function_call("read_file", {"path": "sample.txt"})
    assert "Malformed tool execution interceptor decision" in result
    assert native == []


def test_callback_exception_fails_closed(monkeypatch):
    def fail(*a, **kw):
        raise RuntimeError("callback failed")
    model_tools, native = _prepare_call(monkeypatch, [fail])
    result = model_tools.handle_function_call("read_file", {"path": "sample.txt"})
    assert "Tool execution interceptor failed" in result
    assert native == []


def test_skip_execution_middleware_still_invokes_final_interceptor(monkeypatch):
    from hermes_cli.plugins import ExecutionDecision
    seen = []
    model_tools, native = _prepare_call(monkeypatch, [lambda name, args, **kw: seen.append((name, args)) or ExecutionDecision.handled("intercepted")])
    result = model_tools.handle_function_call("read_file", {"path": "sample.txt"}, skip_tool_execution_middleware=True)
    assert result == "intercepted"
    assert seen == [("read_file", {"path": "sample.txt"})]
    assert native == []


def test_no_matching_interceptor_preserves_native_execution(monkeypatch):
    model_tools, native = _prepare_call(monkeypatch)
    result = model_tools.handle_function_call("read_file", {"path": "sample.txt"})
    assert result == "native-result"
    assert len(native) == 1
