import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.vectorstores import chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from transformers import AutoModel, AutoTokenizer
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
# from langchain.llms import HuggingFaceHub
import json
import requests
from streamlit_lottie import st_lottie
import os
import chromadb
from pyprojroot import here
from openai import AzureOpenAI

def create_chromaDB(pdf_docs):
    chroma_client=chromadb.PersistentClient(path='resources/chroma')
    azure_client = AzureOpenAI(
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_version=os.getenv('AZURE_OpenAI_API_VERSION'),
        azure_endpoint=os.getenv('AZURE_OPENAI_EMBEDDINGS_ENDPOINT')
    )
    for pdf in pdf_docs:

        try:
            if chroma_client.get_collection(name=pdf):
                print('collection already exists')
                #collection=chroma_client.get_collection(name=pdf)
                break
        except Exception as e:
            collection=chroma_client.create_collection(name=pdf)


        loader=PyPDFLoader(os.path.join('resources/books',pdf))
        docs=loader.load()
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300,
        )
        chunks=text_splitter.split_documents(docs)
        docs=[]
        metadata=[]
        ids=[]
        embeddings=[]
        output_str=''
        for index,chunk in enumerate(chunks):
            # output_str+=f'{chunk},\n'
            response=azure_client.embeddings.create(
                input=chunk,
                model=os.getenv('AZURE_OPENAI_EMBEDDING_MODEL_NAME')
            )
            embeddings.append(response.data[0].embedding)
            docs.append(f'{chunk}\n')
            metadata.append({'source':pdf})
            ids.append(f'id{index}')

        collection.add(
            embeddings=embeddings,
            documents=docs,
            metadatas=metadata,
            ids=ids
        )





if __name__ == '__main__':
    print(f'load environment variables:{load_dotenv()}')
    books=os.listdir('resources/books/')
    create_chromaDB(books)
    chroma_client=chromadb.PersistentClient('resources/chroma')
    vectordb=chroma_client.get_collection(name='Resume2024.pdf')

    query_texts = 'what are the companies he has worked in?'


    azure_client = AzureOpenAI(
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_version=os.getenv('AZURE_OpenAI_API_VERSION'),
        azure_endpoint=os.getenv('AZURE_OPENAI_EMBEDDINGS_ENDPOINT')
    )
    response = azure_client.embeddings.create(
        input=query_texts,
        model='text-embedding-ada-002'
    )
    query_embeddings = response.data[0].embedding

    result=vectordb.query(
        query_embeddings=query_embeddings,
        n_results=1
    )
    system_role = "You will recieve the user's question along with the search results of that question over a database. Give the user the proper answer."
    prompt = f"User's question: {query_texts} \n\n Search results:\n {result}"

    message = [
        {
            'role': 'system', 'content': str(system_role)
        },
        {
            'role': 'user', 'content': prompt
        }
    ]
    llm = AzureOpenAI(
        api_version=os.getenv('AZURE_OpenAI_API_VERSION'),
        azure_endpoint=os.getenv('AZURE_OpenAI_ENDPOINT'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY')
    )
    response = llm.chat.completions.create(
        model=os.getenv('AZURE_OPENAI_DEPLOYMENT_MODEL'),
        messages=message
    )
    print(response.choices[0].message.content)


