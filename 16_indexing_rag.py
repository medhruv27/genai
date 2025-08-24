import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

pdf_path = Path(__file__).parent / "nodejs.pdf"
if not pdf_path.exists():
    raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

# Loading
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()  # Read PDF file page-by-page

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400,
)

split_docs = text_splitter.split_documents(documents=docs)

"""
from langchain_openai import OpenAIEmbeddings

OpenAI Embeddings
==================
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=api_key
)
"""
try:
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",  # Update to correct model name
        google_api_key=api_key
    )
except Exception as e:
    raise Exception(f"Failed to initialize embedding model: {e}")

try:
    vector_store = QdrantVectorStore.from_documents(
        documents=split_docs,
        url="http://localhost:6333",
        collection_name="learning_vectors",
        embedding=embedding_model,
        force_recreate=True
    )
    print("Indexing of Documents Done...")
except Exception as e:
    raise Exception(f"Failed to create vector store: {e}")

print("Indexing of Documents Done...")