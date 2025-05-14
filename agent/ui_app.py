import streamlit as st
from agentic_system import AgenticSystem

st.set_page_config(page_title="Lifesight RAG Q&A", page_icon=":mag:", layout="wide")

st.title("Lifesight Marketing Measurement Q&A")
st.write("Ask any question about the Lifesight marketing measurement documentation.")

if "agent" not in st.session_state:
    st.session_state.agent = AgenticSystem()

question = st.text_input("Enter your question:", "")

if st.button("Get Answer") and question.strip():
    with st.spinner("Retrieving answer..."):
        answer = st.session_state.agent.answer_question(question)
    st.markdown("**Answer:**")
    st.write(answer) 