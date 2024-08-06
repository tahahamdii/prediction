import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from qdrant_client import QdrantClient
from langchain_community.chat_models import ChatOllama


import chainlit as cl
from langchain.chains import RetrievalQA

# bring in our GROQ_API_KEY
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

