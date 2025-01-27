import requests
import json
import qdrant_client
import injest
import state
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
from tqdm import tqdm

client = None

def init(user_input):
    injest.injest_data(None)
    return prompt_engine(user_input)

def prompt_engine(user_input):
    index = None

    #Main Logic 
    if index is None or not state.getFlag():
        index = state.getIndex()
        state.setFlag(False)

    retriever = index.as_retriever()
    print("Retrieving nodes...")
    nodes = list(tqdm(retriever.retrieve(user_input), desc="Nodes Retrieved"))

    relevant_docs = ""
    print("Processing retrieved nodes...")
    for x in tqdm(nodes, desc="Processing Nodes"):
        relevant_docs += f"Reference: {x.node.metadata.get('file_name', 'N/A')}, page: {x.node.metadata.get('page_label', 'N/A')}\n"
        relevant_docs += x.node.text
        relevant_docs += "\n\n"

    full_response = []
    prompt = """System: You are an AI assistant specialized in answering questions using the provided context. 
    Use only the provided context to generate your response. 
    If the context does not contain sufficient information to answer the question, clearly state, "I cannot provide an answer with the available information."
    If the user input asks to perform some form of log analysis, clearly state, "I cannot perform log analysis with my current capability."
    
    The provided context includes references formatted as:
    Reference: [file_name], page: [page_label]
    Text: [context content]

    ---------------------
    {relevant_document}
    ---------------------

    Task:
    1. Answer the user’s question strictly based on the provided context. Do not use any prior knowledge or generate information outside the context.
    2. Include the references from the context that were directly used to answer the question. Use the exact format given below:
    - Reference: [file_name], page: [page_label]
    3. Ensure the references are clearly mentioned at the end of your response.
    4. Append the following disclaimer to the response: 'This is an AI-generated response based on retrieved information; accuracy may vary'

    User: {user_input}

    Helpful Answer:"""

    url = 'http://localhost:11434/api/generate'
    data = {
        "model": "codellama",
        "prompt": prompt.format(user_input=user_input, relevant_document=relevant_docs),
        "stream": True
    }
    headers = {'Content-Type': 'application/json'}

    print("Sending request to API...")
    response = requests.post(url, data=json.dumps(data), headers=headers, stream=True)

    try:
        print("Receiving streamed response...")
        for line in tqdm(response.iter_lines(), desc="Streaming Response"):
            if line:  # filter out keep-alive new lines
                decoded_line = json.loads(line.decode('utf-8'))
                full_response.append(decoded_line['response'])
    finally:
        response.close()

    llm_response = ''.join(full_response)
    print("Response from LLM has been sent")
    return llm_response

def cleanup():
    if client is not None and isinstance(client, qdrant_client.QdrantClient):
        client.close()
