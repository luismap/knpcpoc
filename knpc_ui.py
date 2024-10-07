from copy import deepcopy
import streamlit as st
from streamlit_chat import message
from streamlit.components.v1 import html
from core.utils.ChatHistory import ChatHistory
from core.utils.StreamBot import StreamBot
from core.utils.Reranker import Reranker


st.title("Knowledge base AI🤖")
st.logo("resources/img/knpclogo_400x400.jpg", size="large")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = ChatHistory()
if "streambot" not in st.session_state:
    st.session_state.streambot = StreamBot()
if "reranker" not in st.session_state:
    st.session_state.reranker = Reranker()
if "user_id" not in st.session_state:
    st.session_state.user_id = "default"

user_id = st.sidebar.text_input("👤 username", value=st.session_state.user_id)
st.session_state.user_id = user_id

# Display chat messages from history on app rerun
for message in st.session_state.messages.get_history(user_id=user_id):
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


def current_context_from_history():
    user_id = st.session_state.user_id
    return st.session_state.messages.get_history(user_id=user_id)


chat = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

# React to user input
if prompt := st.chat_input("Ask me something..."):
    stbot = st.session_state.streambot
    reranker = st.session_state.reranker
    # text_context = get_current_chat_context()
    text_context = current_context_from_history()

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
        prompt_context = f"""using the following context answer the question. if the context is too short
            or does not related to the question, then answer that you user needs to expand query to retrieve more context.
            ##context
            {context}
            ##question:
            {prompt}
        """
        # create new copy with enhanced context
        llm_context = deepcopy(text_context)
        llm_context.append({"role": "user", "content": prompt_context})
        # copy just the prompt to the history
        text_context.append({"role": "user", "content": prompt})
        # text_context.append({"role": "user", "content": prompt})
        inputs = stbot.generate_inputs(llm_context)
        with st.spinner("asking llm with enhanced context..."):
            a = st.write_stream(stbot.get_stream_output(inputs, max_new_tokens=10000))
            st.write(footer)
            # st.write(a)
            # st.write(text_context)
            ai_rsp = {"role": "assistant", "content": a}

    # # Add user message to chat history
    # st.session_state.messages.append({"role": "user", "content": prompt})
    # st.session_state.messages.append({"role": "assistant", "content": a})
    user_id = st.session_state.user_id
    st.session_state.messages.add_history(user_id, prompt, ai_rsp)
