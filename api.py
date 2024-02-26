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

import argparse


#llama = LlamaCppEmbeddings(model_path="/workspace/FastChat/vicuna-7b-v1.3/pytorch_model-00001-of-00002.bin")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)

#openai.api_key = "EMPTY" # Not support yet
#openai.api_base = "http://localhost:8000/v1"


HOST = 'localhost:8000'
URI = f'http://{HOST}/v1/chat/completions'

model = "/workspace/FastChat/vicuna-7b-v1.3/"


class VicunaLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if isinstance(stop, list):
            stop = stop + ["\n###","\nObservation:"]

        response = requests.post(
            URI,
            json={
                "temperature": 0,
                "max_tokens": 512,
                "model":model,
                "messages":[{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}
llm = VicunaLLM()

#docs = loader.load_and_split(text_splitter)
#documents = text_splitter.create_documents([augment])
db = FAISS.load_local("ipcc_db_full", LlamaCppEmbeddings(model_path="/workspace/FastChat/vicuna-7B-v1.3-GGML/vicuna-7b-v1.3.ggmlv3.q4_0.bin"))

PROMPT_TEMPLATE = """You are the Climate expert.
Your task is to answer common questions on climate change.
Answer the question with the knowledge you learned.
Please provide short and clear answers. Be polite and helpful.
Context:
{context}

Question:
{question}

Your answer:
"""


prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["question", "context"])
climate_qa_chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever(search_type="similarity", search_kwargs={'k':3}), prompt=prompt)




#qa_chain = load_qa_chain(llm, chain_type="refine")

#qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
def main():
    parser = argparse.ArgumentParser(description='prompt input')
    parser.add_argument('--prompt', type=str, default='Is sea level rise avoidable? When will it stop?',
                        help='Input the prompt')
    t_start = time.time()
    print(climate_qa_chain(parser.parse_args().prompt))
    #print(qa_document_chain.run(input_document=augment, question="What structure is classified as a definite lie algebra?"))
    t_end = time.time() - t_start

    print("duration:", t_end)

if __name__ == "__main__":
    main()

