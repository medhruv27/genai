from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

loader = WebBaseLoader(
    ["https://docs.chaicode.com/youtube/chai-aur-html/welcome/", "https://docs.chaicode.com/youtube/chai-aur-html/introduction/","https://docs.chaicode.com/youtube/chai-aur-html/emmit-crash-course/",
     "https://docs.chaicode.com/youtube/chai-aur-html/html-tags/"]
)
chai_aur_html = loader.load()
print(chai_aur_html[1])
loader = WebBaseLoader(["https://docs.chaicode.com/youtube/chai-aur-git/welcome/","https://docs.chaicode.com/youtube/chai-aur-git/introduction/","https://docs.chaicode.com/youtube/chai-aur-git/terminology/",
                        "https://docs.chaicode.com/youtube/chai-aur-git/behind-the-scenes/","https://docs.chaicode.com/youtube/chai-aur-git/branches/","https://docs.chaicode.com/youtube/chai-aur-git/diff-stash-tags/","https://docs.chaicode.com/youtube/chai-aur-git/managing-history/","https://docs.chaicode.com/youtube/chai-aur-git/github/"])

chai_aur_git=loader.load()
print(chai_aur_git[1])

loader=WebBaseLoader(["https://docs.chaicode.com/youtube/chai-aur-c/welcome/","https://docs.chaicode.com/youtube/chai-aur-c/introduction/","https://docs.chaicode.com/youtube/chai-aur-c/hello-world/","https://docs.chaicode.com/youtube/chai-aur-c/variables-and-constants/",
                      "https://docs.chaicode.com/youtube/chai-aur-c/data-types/","https://docs.chaicode.com/youtube/chai-aur-c/operators/","https://docs.chaicode.com/youtube/chai-aur-c/control-flow/",
                      "https://docs.chaicode.com/youtube/chai-aur-c/loops/","https://docs.chaicode.com/youtube/chai-aur-c/functions/"])

chai_aur_cpp=loader.load()
print(chai_aur_cpp[0])

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # max characters per chunk
    chunk_overlap=300  # how much to overlap between chunks
)

chai_aur_html_splits = splitter.split_documents(documents=chai_aur_html)
chai_aur_cpp_splits = splitter.split_documents(documents=chai_aur_cpp)
chai_aur_git_splits = splitter.split_documents(documents=chai_aur_git)
print(len(chai_aur_cpp_splits))
print(len(chai_aur_git_splits))
print(len(chai_aur_html_splits))



embedder= GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",  # Update to correct model name
    google_api_key=api_key
)


# #vector store
from langchain_qdrant import QdrantVectorStore


vector_store=QdrantVectorStore.from_documents(
documents=[],
url="http://localhost:6333",
collection_name="chai_aur_cpp",#table_name
embedding=embedder #which embedding model to use
)
vector_store.add_documents(documents=chai_aur_cpp_splits)
print("Injestion for chai aur cpp done")

vector_store=QdrantVectorStore.from_documents(
documents=[],
url="http://localhost:6333",
collection_name="chai_aur_git",#table_name
embedding=embedder #which embedding model to use
)
vector_store.add_documents(documents=chai_aur_git_splits)
print("Injestion for chai aur git done")


vector_store=QdrantVectorStore.from_documents(
documents=[],
url="http://localhost:6333",
collection_name="chai_aur_html",#table_name
embedding=embedder #which embedding model to use
)
vector_store.add_documents(documents=chai_aur_html_splits)
print("Injestion for chai aur html done")
