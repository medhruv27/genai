from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = ['You are reading dhruv blog.']

embeddings = model.encode(sentences)
print(embeddings)
