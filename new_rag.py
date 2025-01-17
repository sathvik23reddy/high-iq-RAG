import requests
import json
import qdrant_client
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
collection_name="AllIncKB"

def init():
    client = qdrant_client.QdrantClient(
        host="localhost",
        port=6333
    )

    lc_embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embed_model = LangchainEmbedding(lc_embed_model)

    collection_exists = client.collection_exists(collection_name=collection_name)
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    if collection_exists:
        #Proceed to fetch
        pass
    else:
        #Injest data
        injest_data(vector_store, embed_model)

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    prompt_engine(index)


def injest_data(vector_store, embed_model):
    reader = SimpleDirectoryReader("./data/" , recursive=True)
    documents = reader.load_data(show_progress=True)
    pipeline = IngestionPipeline(
        transformations=[
            # MarkdownNodeParser(include_metadata=True),
            # TokenTextSplitter(chunk_size=500, chunk_overlap=20),
            SentenceSplitter(chunk_size=1024, chunk_overlap=20),
            # SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95 , embed_model=Settings.embed_model),
            embed_model,
        ],
        vector_store=vector_store,
    )
    nodes = pipeline.run(documents=documents , show_progress=True)
    print("Number of chunks added to vector DB :",len(nodes))

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
    prompt = """System: You are a AI assistant who is well versed with answering questions from the provided context. Always answer the question, even if the context isn't helpful

    "Context information is below.\n"
    "---------------------\n"
    "{relevant_document}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question from user"
    "User: {user_input}\n"

    Helpful Answer:"""
    url = 'http://localhost:11434/api/generate'
    data = {
        "model": "llama3.2",
        "prompt": prompt.format(user_input=user_input, relevant_document=relevant_docs)
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