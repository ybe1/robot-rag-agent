from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime

from agent.tools.middleware import monitor_tool


def make_request(count=0, name="test_tool"):
    runtime = Runtime(
        context={
            "tool_call_count": count,
            "report": False,
        }
    )
    request = ToolCallRequest(
        tool_call={
            "name": name,
            "args": {},
            "id": "call-1",
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=runtime,
    )
    return request


def test_sixth_tool_call_is_blocked():
    request = make_request(count=5)
    handler_called = False

    def handler(_request):
        nonlocal handler_called
        handler_called = True

    result = monitor_tool.wrap_tool_call(request, handler)

    assert handler_called is False
    assert result.status == "error"
    assert "达到上限" in result.content


def test_tool_exception_becomes_friendly_error_message():
    request = make_request()

    def failing_handler(_request):
        raise RuntimeError("内部敏感错误")

    result = monitor_tool.wrap_tool_call(request, failing_handler)

    assert result.status == "error"
    assert result.content == "工具暂时不可用，请稍后重试。"
    assert "内部敏感错误" not in result.content
    assert request.runtime.context["tool_call_count"] == 1


def test_report_tool_updates_report_context():
    request = make_request(name="fill_context_for_report")

    result = monitor_tool.wrap_tool_call(
        request,
        lambda _request: ToolMessage(
            content="ok",
            tool_call_id="call-1",
        ),
    )

    assert result.content == "ok"
    assert request.runtime.context["report"] is True
