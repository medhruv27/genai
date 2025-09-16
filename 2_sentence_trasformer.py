# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('all-MiniLM-L6-v2')
# sentences = ['You are reading dhruv blog.']

# embeddings = model.encode(sentences)
# print(embeddings)

import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

text = "dog chases cat"

response = client.embeddings.create(
    model="text-embedding-004",
    input=text
)

print("Vector Embeddings", response)

# Indicator for no. of dimensions for vector embedding
print("Length", len(response.data[0].embedding))