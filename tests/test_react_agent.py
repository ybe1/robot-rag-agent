from langchain_core.messages import AIMessage, ToolMessage

from agent.react_agent import ReactAgent


class FakeAgent:
    def __init__(self, messages):
        self.messages = messages
        self.last_context = None

    def invoke(self, _input, context):
        self.last_context = context
        return {"messages": self.messages}


def make_react_agent(messages):
    agent = ReactAgent.__new__(ReactAgent)
    agent.agent = FakeAgent(messages)
    return agent


def test_agent_collects_and_deduplicates_tool_citations():
    citation = {
        "id": 1,
        "source": "故障排除.txt",
        "page": None,
        "content": "检查电源。",
    }
    messages = [
        ToolMessage(
            content="资料回答",
            tool_call_id="call-1",
            artifact={"citations": [citation]},
        ),
        ToolMessage(
            content="重复资料回答",
            tool_call_id="call-2",
            artifact={"citations": [citation]},
        ),
        AIMessage(content="最终回答。[资料1]"),
    ]
    agent = make_react_agent(messages)

    result = agent.execute_with_details([{"role": "user", "content": "测试"}])

    assert result == {
        "answer": "最终回答。[资料1]",
        "citations": [citation],
    }
    assert agent.agent.last_context == {
        "report": False,
        "tool_call_count": 0,
    }


def test_execute_keeps_original_string_return_type():
    agent = make_react_agent([AIMessage(content="普通回答")])

    result = agent.execute([{"role": "user", "content": "测试"}])

    assert result == "普通回答"
