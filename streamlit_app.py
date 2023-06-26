import os
import tempfile
import streamlit as st
import pandas as pd
from io import StringIO
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms.openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


st.set_page_config(page_title="CSV AI", layout="wide")


def home_page():
    st.write("Select any one feature from above sliderbox: \n"
             "1. Chat with CSV \n"
             "2. Summarize CSV (Future) \n"
             "3. Analyze CSV (Future)")


def chat(temperature, model_name):
    reset = st.sidebar.button("Reset Chat")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV here 👇:", type="csv")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        try:
            loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
            data = loader.load()
        except:
            loader = CSVLoader(file_path=tmp_file_path, encoding="cp1252")
            data = loader.load()

        embeddings = OpenAIEmbeddings()
        vectors = FAISS.from_documents(data, embeddings)
        llm = ChatOpenAI(temperature=temperature, model_name=model_name)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                         retriever=vectors.as_retriever(),
                                         verbose=True)

        def conversational_chat(query):
            result = qa.run(query)
            st.session_state['history'].append((query, result))
            return result

        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generatedd' not in st.session_state:
            st.session_state['generatedd'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

        if 'pastt' not in st.session_state:
            st.session_state['pastt'] = ["Hey ! 👋"]

        response_container = st.container()
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
                submit_button = st.form_submit_button(label='Send')

                if submit_button and user_input:
                    output = conversational_chat(user_input)
                    st.session_state['pastt'].append(user_input)
                    st.session_state['generatedd'].append(output)

        if st.session_state['generatedd']:
            with response_container:
                for i in range(len(st.session_state['generatedd'])):
                    message(st.session_state["pastt"][i], is_user=True, key=str(i) + '_user', avatar_style="fun-emoji")
                    message(st.session_state["generatedd"][i], key=str(i), avatar_style="bottts")
        if reset:
            st.session_state["pastt"] = []
            st.session_state["generatedd"] = []



# Main App
def main():
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>Understand-CSV</h1>
            <h3>Interact and Summarize your CSV data using AI!</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    select_box = st.sidebar.selectbox("Select Feature", ("Home", "Chat", "Summarize", "Analyze"), index=0)
    temperature = st.sidebar.slider('Temperature:', min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    model_name = st.sidebar.text_input("Model Name:", "text-davinci-003")

    if select_box == "Home":
        home_page()
    elif select_box == "Chat":
        chat(temperature, model_name)
    

if __name__ == "__main__":
    main()
