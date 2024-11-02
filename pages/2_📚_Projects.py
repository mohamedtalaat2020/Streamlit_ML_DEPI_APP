import streamlit as st


st.title("Projects")
st.sidebar.success("Select a page above.")
#my_input = st.text_input("Input a text here", st.session_state["my_input"])
#st.session_state["my_input"] = my_input
st.write("You have entered", st.session_state["my_input"])