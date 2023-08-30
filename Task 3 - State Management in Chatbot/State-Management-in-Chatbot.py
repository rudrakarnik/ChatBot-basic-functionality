import streamlit as st
import random
from dotenv import load_dotenv
import openai
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import MongoDBChatMessageHistory



def main():
    load_dotenv()

    def _get_session():
        session_id = get_script_run_ctx().session_id
        return session_id
    
    user_session = _get_session()

    connection_string = "YOUR_CONNECTION_STRING"
    message_history = MongoDBChatMessageHistory(
    connection_string=connection_string, session_id = user_session)

    st.header("Chat with PDF 📃")
    
    pdf = st.file_uploader("Upload your PDF", type='pdf')        

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text=""
        
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
        )
        
        chunks=text_splitter.split_text(text)
    
        embeddings=OpenAIEmbeddings()
        Vectorstore = FAISS.from_texts(chunks,embedding=embeddings)
        
        llm = OpenAI(temperature=0)

        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
        retriever = Vectorstore.as_retriever()


        qachat = ConversationalRetrievalChain.from_llm(
                llm=llm,
                memory=memory,
                retriever=retriever  
            )
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if query := st.chat_input("Ask questions?"):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            with get_openai_callback() as cb:
                response=qachat({"question":query})
                print(cb)
            result=response["answer"]
            with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})    
            message_history.add_user_message(response["question"])
            message_history.add_ai_message(response["answer"])


if __name__ == "__main__":
    main()