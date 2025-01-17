import qdrant_client
import os
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

def get_index(vector_store, embed_model):
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    return index

def injest_data(path):
    client = qdrant_client.QdrantClient(
        host="localhost",
        port=6333
    )

    lc_embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embed_model = LangchainEmbedding(lc_embed_model)

    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)

    reader = None
    if path is None:
        return get_index(vector_store, embed_model)
    elif os.path.isfile(path):
        reader = SimpleDirectoryReader(input_files=[path] , recursive=True)
    elif os.path.isdir(path):
        reader = SimpleDirectoryReader(input_dir=path , recursive=True)
    else:
        reader = SimpleDirectoryReader(input_dir="./data/" , recursive=True)

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

    return get_index(vector_store, embed_model)

def main():
    path = input("Provide file path/dir to train RAG: ")
    injest_data(path)

if __name__=="__main__":
    main()