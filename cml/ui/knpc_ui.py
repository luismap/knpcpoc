import streamlit as st
from streamlit_chat import message
from streamlit.components.v1 import html

from StreamBot import StreamBot
from Reranker import Reranker

import sys

sys.path.append("../../")

st.title("KNPC Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "streambot" not in st.session_state:
    st.session_state.streambot = StreamBot()
if "reranker" not in st.session_state:
    st.session_state.reranker = Reranker()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def get_llm_context(top_docs_list):
    """given a tuple of (ids, docs, metas)
    return a formatted context for the llm
    """
    context = "\n\n".join([e[1] for e in top_docs_list])
    return context


def get_source_info(top_docs_list):
    """given a tuple of (ids, docs, metas)
    return a formatted source info
    """
    source_info = "\n".join(
        [
            "* page: " + str(e[2]["page"]) + ",doc: " + e[2]["source"].split("/")[-1]
            for e in top_docs_list
        ]
    )

    return "*sources*:\n" + source_info


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
    reranker = st.session_state.reranker
    text_context = get_current_chat_context()

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        with st.spinner("quering reranker and getting best docs..."):
            reranker_payload = {
                "question": prompt,
                "n_gen": 5,
                "n_doc": 4,
                "top_n": 4,
                "embeddings": 0,
            }
            top_docs = reranker.rerank(reranker_payload)
    with st.chat_message("assistant"):
        context = get_llm_context(top_docs)
        footer = get_source_info(top_docs)
        prompt_context = f"""using the following context
            ##context
            {context}
            answer this question:
            {prompt}
        """
        text_context.append({"role": "user", "content": prompt_context})
        inputs = stbot.generate_inputs(text_context)
        with st.spinner("asking llm with enhanced context..."):
            # st.write(a)
            a = st.write_stream(stbot.get_stream_output(inputs, max_new_tokens=10000))
            st.write(footer)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": a})
