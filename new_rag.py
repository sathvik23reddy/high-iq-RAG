from llama_index.llms.langchain import LangChainLLM
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.ingestion import IngestionPipeline
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core import ChatPromptTemplate
from IPython.display import Markdown, display
import qdrant_client
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Document


reader = SimpleDirectoryReader("./data/" , recursive=True)
documents = reader.load_data(show_progress=True)

client = qdrant_client.QdrantClient(
    host="localhost",
    port=6333
)

vector_store = QdrantVectorStore(client=client, collection_name="BASIC_RAG")

lc_embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
embed_model = LangchainEmbedding(lc_embed_model)

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
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

nodes = pipeline.run(documents=documents , show_progress=True)
print("Number of chunks added to vector DB :",len(nodes))

qa_prompt_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {query_str}\n"
)

refine_prompt_str = (
    "We have the opportunity to refine the original answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question: {query_str}. "
    "If the context isn't useful, output the original answer again.\n"
    "Original Answer: {existing_answer}"
)

chat_text_qa_msgs = [
    ("system","You are a AI assistant who is well versed with answering questions from the provided context"),
    ("user", qa_prompt_str),
]
text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

chat_refine_msgs = [
    ("system","Always answer the question, even if the context isn't helpful.",),
    ("user", refine_prompt_str),
]
refine_template = ChatPromptTemplate.from_messages(chat_refine_msgs)

llm = MockLLM()

BASE_RAG_QUERY_ENGINE = index.as_query_engine(
        similarity_top_k=5,
        text_qa_template=text_qa_template,
        refine_template=refine_template,
        llm=llm)

response = BASE_RAG_QUERY_ENGINE.query("Ask a question")
display(Markdown(str(response)))