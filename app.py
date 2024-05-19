import streamlit as st
import time
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

# loading Groq and OpenAI API keys
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

st.title("ObjectBox Vector Store DB with Llama3 Demo")

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """

)

## Vector Embedding and ObjectBox Vectorstore DB

def vector_embedding():
    
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OpenAIEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("./us_census") # data ingestion
        st.session_state.docs=st.session_state.loader.load() # loading documents
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors=ObjectBox.from_documents(st.session_state.final_documents, st.session_state.embeddings, embedding_dimensions=768)


input_prompt=st.text_input("Ask questions on the uploaded documents...")

if st.button("Click to start document embeddings"):
    vector_embedding()
    st.write("ObjectBox Database is ready")
    

if input_prompt:
    document_chain=create_stuff_documents_chain(llm, prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever, document_chain)
    start=time.process_time()
    
    response=retrieval_chain.invoke({"input": input_prompt})
    print("Response time: ", time.process_time()-start)
    st.write(response["answer"])
    
    # Streamlit Expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-------------------------")