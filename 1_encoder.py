#converts word to tokens
import tiktoken

encoder=tiktoken.encoding_for_model("gpt-4o")
text="Effiel tower is the highest building in the world"
encoded_text=encoder.encode(text)
print(encoded_text)
decoded_text=encoder.decode(encoded_text)
print(decoded_text)
encoder.n_vocab #gives vocab size 2,00,019(no of unique tokens)
print(encoder.n_vocab)