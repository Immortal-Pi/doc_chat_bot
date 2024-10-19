import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModel, AutoTokenizer
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
from langchain_chroma import Chroma
from chromadb import Client
# from langchain.llms import HuggingFaceHub
import json
import requests
from streamlit_lottie import st_lottie
import os
import chromadb
from pyprojroot import here
from prepare_vectordb_from_text_chunks import create_chromaDB
import warnings
import pandas as pd

run_once=0
def load_lottiefile(filepath:str):
    with open(filepath,'r',encoding='utf-8') as f:
        return json.load(f)

def load_lottieurl(url:str):
    r=requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()


def create_chromaDB(pdf_docs):
    chroma_client=chromadb.PersistentClient(path=str(here('resources/chroma')))

    create_chromaDB(pdf_docs)



def get_pdf_text(pdf_docs):
    text=""

    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter=CharacterTextSplitter(
        separator='\n',
        chunk_size=1500,
        chunk_overlap=400,
        length_function=len
    )
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunk):
    # OPEN AI embeddings technique and FAISS vector store
    embeddings=OpenAIEmbeddings()
    index_path='resources/FAISS/'
    vectorstore=FAISS.from_texts(texts=text_chunk,embedding=embeddings)
    if os.path.exists('resources/FAISS/index.faiss'):
        store = FAISS.load_local('resources/FAISS/', embeddings, allow_dangerous_deserialization=True)
        store.merge_from(vectorstore)
        store.save_local(index_path)
        return store
    else:
        vectorstore.save_local(index_path)
        return vectorstore

def get_conversation_chain():
    embeddings = OpenAIEmbeddings()
    # collection=
    vectorstore = FAISS.load_local('resources/FAISS/', embeddings, allow_dangerous_deserialization=True)
    # vectorstore=Chroma(
    #     collection_name=collection,
    #     embedding_function=embeddings
    # )
    llm=ChatOpenAI()
    # llm=HuggingFaceHub(repo_id='nvidia/Llama-3_1-Nemotron-51B-Instruct',model_kwargs={"temperature":0.5,"max_length":512})
    memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    return conversation_chain

def handle_userinput(user_input):
    # embeddings = OpenAIEmbeddings()
    # llm = ChatOpenAI()
    response=st.session_state.conversation({'question':user_input})
    st.session_state.chat_history=response['chat_history']
    # vectorstore = FAISS.load_local('resources/FAISS/index.index', embeddings)
    # st.write(vectorstore)
    # memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    # conversation_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=vectorstore.as_retriever(),
    #     memory=memory
    # )
    for i, message in enumerate(reversed(st.session_state.chat_history)):
        if i%2==0:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            # st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
        else:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            # st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
    # st.write(response)

def print_existing_books():
    #embeddings = OpenAIEmbeddings()
    #vectorstore = FAISS.load_local('resources/FAISS/', embeddings, allow_dangerous_deserialization=True)
    df=pd.read_csv('resources/books/all.csv')
    st.write(df['books'])


if __name__=='__main__':
    #warnings.filterwarnings('ignore')
    lottie_books=load_lottieurl('https://lottie.host/262a2841-5ec5-4228-9e10-f1c40368652c/Mc4Pv7r5p5.json')
    lottie_books_local=load_lottiefile('resources/books.json')
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=[]

    load_dotenv()

    st.set_page_config(page_title='Chat with multiple PDF')
    st_lottie(
        lottie_books,
        speed=1,
        reverse=True,
        loop=False,
        quality='medium',
        # renderer='svg',
        height=300,
        width=300,
        key=None,
        
    )
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with PDF's")
    user_question=st.text_input("Ask me anything about the documents:")
    
    
    #action=run_once(get_session_state())
    
    if user_question:
        handle_userinput(user_question)

    # st.write(user_template.replace("{{MSG}}","hello"),unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}","hello "),unsafe_allow_html=True)


    with st.sidebar:
        st.subheader('Your Documents')
        pdf_docs = st.file_uploader("Upload PFS's here and click on process", accept_multiple_files=True)
        docs=[]
        for doc in pdf_docs:
            docs.append(doc.name)
        data_frame=pd.DataFrame({'books':docs})
        print(data_frame)
        if st.button('Process'):
            with st.spinner('processing'):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                #st.write(raw_text)
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # create vector store
                #st.write(text_chunks)
                vectorstore = get_vector_store(text_chunks)
                #st.write(vectorstore)
                #create_chromaDB(pdf_docs)
                print(os.listdir('resources/books'))
                if len(os.listdir('resources/books'))==0:
                    data_frame.to_csv('resources/books/all.csv',header=['books'],index=False)
                else:
                    data_frame.to_csv('resources/books/all.csv',mode='a',index=False,header=False)


                # create conversation chain
                st.session_state.conversation = get_conversation_chain()
        if st.button('From existing books'):
            print_existing_books()
            st.session_state.conversation = get_conversation_chain()




