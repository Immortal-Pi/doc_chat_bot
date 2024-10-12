import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
import chromadb
import json
import requests
import os
import pandas as pd
from pyprojroot import here
from streamlit_lottie import st_lottie
from openai import AzureOpenAI
from langchain_community.embeddings import AzureOpenAIEmbeddings

def load_lottiefile(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    # Create a Chroma vector store
    embeddings_fun = OpenAIEmbeddings()
    # embeddings_fun = AzureOpenAIEmbeddings()
    docs = []
    metadatas = []
    ids = []
    embeddings = []
    chroma_client = chromadb.PersistentClient(path='resources/Chroma')
    collections=chroma_client.get_or_create_collection(name='all')
    azure_client = AzureOpenAI(
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_version=os.getenv('AZURE_OpenAI_API_VERSION'),
        azure_endpoint=os.getenv('AZURE_OPENAI_EMBEDDINGS_ENDPOINT')
    )
    for index,text in enumerate(text_chunks):
        output_str=f'{index}: {text},\n'
        response=azure_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embeddings.append(response.data[0].embedding)
        docs.append(output_str)
        metadatas.append({'source':'all'})
        ids.append(f'id{index}')
    collections.add(
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings,
        ids=ids
    )
    # vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings, client=chroma_client)

    vectorstore = Chroma(client=chroma_client, collection_name="all", embedding_function=embeddings_fun)
    return vectorstore


def get_conversation_chain():
    embeddings = OpenAIEmbeddings()
    # embeddings=AzureOpenAIEmbeddings(
    #     openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    #     endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
    # )
    chroma_client = chromadb.PersistentClient(path='resources/Chroma')
    vectorstore = Chroma(client=chroma_client, collection_name="all", embedding_function=embeddings)

    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain


def handle_userinput(user_input):
    response = st.session_state.conversation({'question': user_input})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(reversed(st.session_state.chat_history)):
        if i % 2 == 0:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def print_existing_books():
    df = pd.read_csv('resources/books/all.csv')
    st.write(df['books'])


if __name__ == '__main__':
    lottie_books = load_lottieurl('https://lottie.host/c9bd90fb-0fb2-4e89-bd8d-af342c6c5d84/bqNbcDkLb9.json')
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    load_dotenv()

    st.set_page_config(page_title='Chat with multiple PDF')
    st_lottie(
        lottie_books,
        speed=1,
        reverse=True,
        loop=False,
        quality='high',
        height=300,
        width=300,
        key=None,
    )
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with PDF's")
    user_question = st.text_input("Ask me anything about the documents:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader('Your Documents')
        pdf_docs = st.file_uploader("Upload PDFs here and click on process", accept_multiple_files=True)
        docs = [doc.name for doc in pdf_docs]
        data_frame = pd.DataFrame({'books': docs})

        if st.button('Process'):
            with st.spinner('Processing'):
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)
                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # Create vector store
                vectorstore = get_vector_store(text_chunks)

                # Save the list of uploaded books
                if len(os.listdir('resources/books')) == 0:
                    data_frame.to_csv('resources/books/all.csv', header=['books'], index=False)
                else:
                    data_frame.to_csv('resources/books/all.csv', mode='a', index=False, header=False)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain()

        if st.button('From existing books'):
            print_existing_books()
            st.session_state.conversation = get_conversation_chain()
