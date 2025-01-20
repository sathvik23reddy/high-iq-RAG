import requests
import json
import qdrant_client
import injest
from transformers import pipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.ingestion import IngestionPipeline
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore

client, index = None, None

def init():
    index = injest.injest_data(None)
    prompt_engine(index)

def prompt_engine(index):
    user_input = """Your question goes here"""

    retriever = index.as_retriever()
    nodes = retriever.retrieve(user_input)

    #Further enhance to use metadata and reference
    relevant_docs = ""
    for x in nodes:
        relevant_docs += x.node.text
        relevant_docs += "\n\n"


    full_response = []
    prompt = """System: You are a AI assistant who is well versed with answering questions from the provided context. 
    "In case the given context isn't helpful, proceed to mention clearly that you cannot help with the available information"
    "Do not generate answers irrelevant to the context\n\n"
    "Context information is below.\n"
    "---------------------\n"
    "{relevant_document}\n"
    "---------------------\n"
    "Given the context information and no prior knowledge"
    "Answer the question from user"
    "User: {user_input}\n"

    Helpful Answer:"""
    url = 'http://localhost:11434/api/generate'
    data = {
        "model": "codellama",
        "prompt": prompt.format(user_input=user_input, relevant_document=relevant_docs),
        "stream": True 
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers, stream=True)
    try:
        for line in response.iter_lines():
            # filter out keep-alive new lines
            if line:
                decoded_line = json.loads(line.decode('utf-8'))
                full_response.append(decoded_line['response'])
    finally:
        response.close()
    print(''.join(full_response))

def cleanup():
    if client is not None and isinstance(client, qdrant_client.QdrantClient):
        client.close()

def main():
    init()

if __name__=="__main__":
    main()