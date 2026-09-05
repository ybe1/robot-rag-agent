import streamlit as st

from agent.react_agent import ReactAgent
from utils.logger_handler import logger


st.title("智扫通机器人智能客服")
st.divider()


# 每个浏览器会话分别保存自己的 Agent 和聊天记录。
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []


# 清空当前会话。
if st.sidebar.button("新建会话"):
    st.session_state["message"] = []
    st.session_state["agent"] = ReactAgent()
    st.rerun()


# 展示已经完成的历史对话。
for message in st.session_state["message"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])


prompt = st.chat_input("请输入你的问题")


if prompt and prompt.strip():
    user_message = {
        "role": "user",
        "content": prompt.strip(),
    }

    with st.chat_message("user"):
        st.write(user_message["content"])

    # 复制历史，再加上当前问题。
    # 先不修改已保存的记录，避免失败请求污染后续对话。
    messages = [
        dict(message)
        for message in st.session_state["message"]
    ]
    messages.append(user_message)

    try:
        with st.spinner("智能客服思考中……"):
            answer = st.session_state["agent"].execute(messages)

    except Exception:
        logger.exception("生成回答失败")
        st.error("本次回答失败，请重新输入问题再试。")

    else:
        # 成功后，再把这一轮问答一起保存。
        st.session_state["message"].extend([
            user_message,
            {
                "role": "assistant",
                "content": answer,
            },
        ])

        with st.chat_message("assistant"):
            st.write(answer)