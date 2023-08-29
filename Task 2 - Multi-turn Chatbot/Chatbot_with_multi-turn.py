import streamlit as st
from dotenv import load_dotenv
import pickle
import openai
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain



def main():
    load_dotenv()

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

            response=qachat({"question":query})
            result=response["answer"]
            with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})    
        
if __name__ == "__main__":
    main()