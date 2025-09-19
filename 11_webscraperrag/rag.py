import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

os.environ["GRPC_VERBOSITY"] = "NONE"
api_key = os.getenv("GOOGLE_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

embedder = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=api_key
)

relevant_chunks=''
System_Prompt="You are an smart ai assitant that basically takes a user query and check whether that user query is related to c++,git or Html on the basis of user query  you have to return one word answer wether the query is related to html or c++ or git on the basis of that return one word answer if query is related to git then return git if it is related to html return html if it is related to c++ return cpp if the query is not related to any of these topic return sorry but i can only help you with queries related to html git and c++"

user_query=input("> ")

response=client.chat.completions.create(
    model='gemini-2.0-flash',
    messages=[{"role":"system","content":System_Prompt},{"role":"user","content":user_query}]
)

print(response.choices[0].message.content)
if(response.choices[0].message.content== "html\n"):
    vector_db=QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name="chai_aur_html",#table_name
        embedding=embedder
    )
    relevant_chunks=vector_db.similarity_search(
        query=user_query
    )
elif(response.choices[0].message.content=="git\n") :   
    vector_db=QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name="chai_aur_git",#table_name
        embedding=embedder
    )
    relevant_chunks=vector_db.similarity_search(
       query=user_query   
    )
elif(response.choices[0].message.content=="cpp\n") :   
    vector_db=QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name="chai_aur_cpp",#table_name
        embedding=embedder
    )
    relevant_chunks=vector_db.similarity_search(
      query=user_query
    )
else:
    print(response.choices[0].message.content)
    exit
# print(relevant_chunks)    
# context = "\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}" for result in relevant_chunks])

Main_prompt=f'''You are a smart ai your work is you take the user query as an input and based on {relevant_chunks} you basically provide the answer of user query in detail. User may interested in knowing from where he could learn more so 
please provide the references available in the context along with the query related content present in the context. navigate the user to open the right page to know more.
{relevant_chunks}
'''    

answer=client.chat.completions.create(
    model='gemini-2.0-flash',
    messages=[{"role":"system","content":Main_prompt},{"role":"user","content":user_query}]
)

    
print(answer.choices[0].message.content)