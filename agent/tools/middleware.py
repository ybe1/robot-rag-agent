from typing import Callable
from utils.prompt_loader import load_system_prompts,load_report_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call,before_model,dynamic_prompt,ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger

@wrap_tool_call
def monitor_tool(
    request:ToolCallRequest,
    handler:Callable[[ToolCallRequest],ToolMessage | Command],
)->ToolMessage | Command:
    tool_call_count = request.runtime.context.get('tool_call_count', 0)

    if tool_call_count >= 5:
        logger.warning('[tool monitor]本轮工具调用已达到5次上限')
        return ToolMessage(
            content='本轮工具调用次数已达到上限，无法继续查询。',
            tool_call_id=request.tool_call['id'],
            status='error',
        )

    request.runtime.context['tool_call_count'] = tool_call_count + 1
    logger.info(f'[tool monitor]执行工具：{request.tool_call["name"]}')
    logger.info(f'[tool monitor]传入参数：{request.tool_call["args"]}')

    try:
        result=handler(request)
        logger.info(f'[tool monitor]工具{request.tool_call["name"]}调用成功')

        if request.tool_call["name"]=='fill_context_for_report':
            request.runtime.context['report']=True

        return result

    except Exception:
        logger.exception(f'工具{request.tool_call["name"]}调用失败')
        return ToolMessage(
            content='工具暂时不可用，请稍后重试。',
            tool_call_id=request.tool_call['id'],
            status='error',
        )

@before_model
def log_before_model(
    state:AgentState,
    runtime:Runtime,
):
    logger.info(f'[log_before_model]即将调用模型，带有{len(state["messages"])}条消息')

    logger.debug(f'[log_before_model]{type(state["messages"][-1]).__name__} | {state["messages"][-1].content.strip()}')

    return None

@dynamic_prompt
def report_prompt_switch(request:ModelRequest):
    is_report=request.runtime.context.get('report',False)
    if is_report:
        return load_report_prompts()

    return load_system_prompts()
    
