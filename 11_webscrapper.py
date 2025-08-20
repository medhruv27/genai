from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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


import os
api_key=os.getenv("OPENAI_API_KEY")
from langchain_openai import OpenAIEmbeddings
embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=api_key,
)


# #vector store
from langchain_qdrant import QdrantVectorStore


# vector_store=QdrantVectorStore.from_documents(
# documents=[],
# url="http://localhost:6333",
# collection_name="chai_aur_cpp",#table_name
# embedding=embedder #which embedding model to use
# )
# vector_store.add_documents(documents=chai_aur_cpp_splits)
# print("Injestion for chai aur cpp done")

# vector_store=QdrantVectorStore.from_documents(
# documents=[],
# url="http://localhost:6333",
# collection_name="chai_aur_git",#table_name
# embedding=embedder #which embedding model to use
# )
# vector_store.add_documents(documents=chai_aur_git_splits)
# print("Injestion for chai aur git done")


# vector_store=QdrantVectorStore.from_documents(
# documents=[],
# url="http://localhost:6333",
# collection_name="chai_aur_html",#table_name
# embedding=embedder #which embedding model to use
# )
# vector_store.add_documents(documents=chai_aur_html_splits)
# print("Injestion for chai aur html done")

relevant_chunks=''
System_Prompt="You are an smart ai assitant that basically takes a user query and check whether that user query is related to c++,git or Html on the basis of user query  you have to return one word answer wether the query is related to html or c++ or git on the basis of that return one word answer if query is related to git then return git if it is related to html return html if it is related to c++ return cpp if the query is not related to any of these topic return sorry but i can only help you with queries related to html git and c++"
from openai import OpenAI
client=OpenAI(api_key=api_key)
user_query=input("Please enter your query ")
response=client.chat.completions.create(
    model='gpt-4o',
    messages=[{"role":"system","content":System_Prompt},{"role":"user","content":user_query}]
)
print(response.choices[0].message.content)
if(response.choices[0].message.content=="html"):
    retriever=QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="chai_aur_html",#table_name
    embedding=embedder
    )
    relevant_chunks=retriever.similarity_search(
    query=user_query,
    )
elif(response.choices[0].message.content=="git") :   
    retriever=QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="chai_aur_git",#table_name
    embedding=embedder
    )
    relevant_chunks=retriever.similarity_search(
    query=user_query,
    )
elif(response.choices[0].message.content=="cpp") :   
    retriever=QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="chai_aur_cpp",#table_name
    embedding=embedder
    )
    relevant_chunks=retriever.similarity_search(
    query=user_query,
    )
else:
    print(response.choices[0].content)
    exit
print(relevant_chunks)    
Main_prompt=f'''You are a smart ai your work is you take the user query as an input and based on relevant chunk you basically provide the answer of user query in detail. User may interested in knowing from where he could learn more so 
please provide the references available in the relevant chunks along with the query related content present in the relevant_chunk.
{relevant_chunks}
'''    

answer=client.chat.completions.create(
    model='gpt-4o',
    messages=[{"role":"system","content":Main_prompt},{"role":"user","content":user_query}]
)

    
print(answer.choices[0].message.content)

