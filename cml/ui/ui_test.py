import streamlit as st
from streamlit_chat import message
from streamlit.components.v1 import html
from cml.ui.StreamBot import get_stream_output, generate_inputs


def on_input_change():
    user_input = st.session_state.user_input
    st.session_state.past.append(user_input)
    input = generate_inputs(user_input)
    ans = st.write_stream(get_stream_output(input))
    st.session_state.generated.append({"type": "markdown", "data": ans})


def on_btn_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]


st.session_state.setdefault(
    "past",
    [
        "plan text with line break",
    ],
)
st.session_state.setdefault(
    "generated",
    [
        {"type": "normal", "data": "Line 1 \n Line 2 \n Line 3"},
    ],
)

st.title("AI-Bot for KNPC")

with st.container():
    st.text_input("User Input:", on_change=on_input_change, key="user_input")
    # st.rerun()


chat_placeholder = st.empty()

with chat_placeholder.container():
    st.button("Clear message", on_click=on_btn_click)
    chat_length = len(st.session_state["generated"])
    for i in range(chat_length)[::-1]:
        message(st.session_state["past"][i], is_user=True, key=f"{i}_user")
        message(
            st.session_state["generated"][i]["data"],
            key=f"{i}",
            allow_html=True,
            is_table=True
            if st.session_state["generated"][i]["type"] == "table"
            else False,
        )
