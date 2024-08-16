import streamlit as st
from langchain_core.messages import ChatMessage
import requests

SERVER_URL = "http://127.0.0.1:8000"


def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def get_response(user_input: str):
    res = requests.post(f"{SERVER_URL}/chat", params={
        "user_input": user_input
    })
    return res.text


if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("마이데이터 챗봇 서비스")

with st.sidebar:
    button = st.button("clear")
    if button:
        st.session_state["messages"] = []

print_messages()

user_input = st.chat_input("마이데이터 관련 궁금한 내용을 물어보세요")
if user_input:
    st.chat_message("user").write(user_input)

    with st.spinner(text="waiting.."):
        ai_answer = get_response(user_input)
        with st.chat_message("assistant"):
            container = st.empty()

            container.markdown(ai_answer)

        st.session_state["messages"].append(ChatMessage(role="user", content=user_input))
        st.session_state["messages"].append(ChatMessage(role="assistant", content=ai_answer))
