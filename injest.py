import qdrant_client
import os
import state
from tqdm import tqdm
from qdrant_client.models import VectorParams, Distance
from qdrant_client import QdrantClient, models
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
collection_name="AllIncKB1"

def placeholder_populate_filemap():
    #Will be replaced by DB in future
    directory = "./data/"
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            state.shared_state['file_map'].add(file)

def set_index(vector_store, embed_model):
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    state.setIndex(index)
    state.setFlag(True)

def removeFromVDB(vector_store, file):
    print(f"Deleting file that exists: {file}")
    # File exists in the map, remove from vector DB if it exists
    vector_store.client.delete(collection_name=collection_name, points_selector=models.FilterSelector(
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="file_name",
                    match=models.MatchValue(value=file),
                ),
            ],
        )
    ))    
    print(f"Removed old data for file: {file}")

def removeFileIfInVDB(vector_store, path):
    file = os.path.basename(path)
    if file in state.shared_state['file_map']:
        removeFromVDB(vector_store, file)

def removeDirIfInVDB(vector_store, dir):
    for file in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, file)):
            removeFileIfInVDB(vector_store, file)


def injest_data(path):
    client = qdrant_client.QdrantClient(
        host="localhost",
        port=6333
    )

    lc_embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embed_model = LangchainEmbedding(lc_embed_model)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name,
            vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE,
            ),
        )

    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)

    reader = None

    if path is None:
        set_index(vector_store, embed_model) 
        return 
    elif os.path.isfile(path):
        reader = SimpleDirectoryReader(input_files=[path], recursive=True)
        removeFileIfInVDB(vector_store, path)
    elif os.path.isdir(path):
        reader = SimpleDirectoryReader(input_dir=path, recursive=True)
        removeDirIfInVDB(vector_store, path)
    else:
        reader = SimpleDirectoryReader(input_dir="./data/", recursive=True)
        removeDirIfInVDB(vector_store, "./data/")

    # Show progress when loading documents
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

    # Wrap the documents in tqdm for progress tracking during the pipeline run
    with tqdm(total=len(documents), desc="Processing documents") as pbar:
        nodes = []
        for document in documents:
            file_name = os.path.basename(document.metadata.get("file_path", "unknown"))
                
            new_nodes = pipeline.run(documents=[document], show_progress=False)
            nodes.extend(new_nodes)

            state.insertFileMap(file_name)

            pbar.update(1)

    print("Number of chunks added to vector DB:", len(nodes))
    set_index(vector_store, embed_model)

def main():
    path = input("Provide file path/dir to train RAG: ").strip()
    placeholder_populate_filemap() #Will be replaced by DB in future
    injest_data(path)

if __name__=="__main__":
    main()