import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from agent.tools.agent_tools import (
    rag_summarize,
    get_weather,
    get_user_location,
    get_user_id,
    get_current_month,
    fetch_external_data,
    fill_context_for_report,
)
from agent.tools.middleware import (
    monitor_tool,
    log_before_model,
    report_prompt_switch,
)
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[
                rag_summarize,
                get_weather,
                get_user_location,
                get_user_id,
                get_current_month,
                fetch_external_data,
                fill_context_for_report,
            ],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute(self, messages: list[dict]) -> str:
        """接收会话历史，返回本轮最终回答。"""

        result = self.agent.invoke(
            {"messages": messages},
            context={"report": False},
        )

        result_messages = result.get("messages", [])
        if not result_messages:
            raise RuntimeError("Agent 没有返回消息")

        final_message = result_messages[-1]

        # 只接收最终 AI 回答，避免把工具结果当作回答。
        if not isinstance(final_message, AIMessage) or final_message.tool_calls:
            raise RuntimeError("Agent 未返回完整的最终回答")

        content = final_message.content

        # 兼容字符串内容和文本块列表。
        if isinstance(content, str):
            answer = content
        else:
            parts = []

            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))

            answer = "".join(parts)

        answer = answer.strip()

        if not answer:
            raise RuntimeError("模型返回了空回答")

        return answer


if __name__ == "__main__":
    agent = ReactAgent()

    messages = [
        {
            "role": "user",
            "content": "帮我生成我的使用报告",
        }
    ]

    answer=agent.execute(messages)
    print(answer)
