import streamlit as st
from streamlit_chat import message
from streamlit.components.v1 import html
from StreamBot import StreamBot

st.title("KNPC Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "streambot" not in st.session_state:
    st.session_state.streambot = StreamBot()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def get_current_chat_context():
    chat_ctx = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            chat_ctx.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            chat_ctx.append({"role": "assistant", "content": m["content"]})
    return chat_ctx


chat = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

# React to user input
if prompt := st.chat_input("What is up?"):
    stbot = st.session_state.streambot
    # Display user message in chat message container
    text_context = get_current_chat_context()
    with st.chat_message("user"):
        text_context.append({"role": "user", "content": prompt})
        st.markdown(prompt)
    with st.chat_message("assistant"):
        inputs = stbot.generate_inputs(text_context)
        a = st.write_stream(stbot.get_stream_output(inputs, max_new_tokens=10000))

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": a})
