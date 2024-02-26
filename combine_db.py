import openai
import time
import csv
import transformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain import hub
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import requests

from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import LlamaCppEmbeddings

#llama = LlamaCppEmbeddings(model_path="/workspace/FastChat/vicuna-7b-v1.3/pytorch_model-00001-of-00002.bin")
loader = PyPDFLoader(

    "https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_Chapter14.pdf"
)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    add_start_index=True,
)

docs = loader.load_and_split(text_splitter)
#documents = text_splitter.create_documents([augment])
db = FAISS.load_local("ipcc_db_tech", LlamaCppEmbeddings(model_path="/workspace/FastChat/vicuna-7B-v1.3-GGML/vicuna-7b-v1.3.ggmlv3.q4_0.bin"))
db_new = FAISS.from_documents(docs, LlamaCppEmbeddings(model_path="/workspace/FastChat/vicuna-7B-v1.3-GGML/vicuna-7b-v1.3.ggmlv3.q4_0.bin"))
db.merge_from(db_new)

db.save_local("ipcc_db_full")