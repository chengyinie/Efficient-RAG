import os

os.environ["OPENAI_API_KEY"] = "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"

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import argparse

# Load the IPCC Climate Change Synthesis Report from a PDF file
loader = PyPDFLoader(
    "https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_LongerReport.pdf"

)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)

# Load the splitted fragments in our vector store
docs = loader.load_and_split(text_splitter)
db = FAISS.load_local("na_db", OpenAIEmbeddings())
#db = FAISS.from_documents(docs, OpenAIEmbeddings())

# We use a simple prompt
PROMPT_TEMPLATE = """You are the Climate Assistant, a helpful AI assistant made by Giskard.
Your task is to answer common questions on climate change.
You will be given a question and relevant excerpts from the IPCC Climate Change Synthesis Report (2023).
Please provide short and clear answers based on the provided context. Be polite and helpful.

Context:
{context}

Question:
{question}

Your answer:
"""

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["question", "context"])
climate_qa_chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever(), prompt=prompt)


def main():
    parser = argparse.ArgumentParser(description='prompt input')
    parser.add_argument('--prompt', type=str, default='Is sea level rise avoidable? When will it stop?',
                        help='Input the prompt')
    print(climate_qa_chain(parser.parse_args().prompt))
    #print(qa_document_chain.run(input_document=augment, question="What structure is classified as a definite lie algebra?"))


if __name__ == "__main__":
    main()
