import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Sample data
# context = "Marie Curie conducted pioneering research on radioactivity."
# query = "What did Marie Curie discover?"
# response = "Marie Curie discovered radium and polonium."

context = "Thomas Alva Edison Invented Bulb"
query = "What did Thomas Alva Edison Invent?"
response = "Thomas Alva Edison Invented Bulb"

# Vectorize texts for cosine similarity
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([query, response, context])
query_vector, response_vector, context_vector = vectors.toarray()

# Cosine Similarity
query_response_similarity = cosine_similarity([query_vector], [response_vector])[0][0]
context_response_similarity = cosine_similarity([context_vector], [response_vector])[0][0]

print(f"Cosine Similarity (Query vs. Response): {query_response_similarity}")
print(f"Cosine Similarity (Context vs. Response): {context_response_similarity}")

# Perplexity
def calculate_perplexity(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()

response_perplexity = calculate_perplexity(response, model, tokenizer)
print(f"Perplexity of Response: {response_perplexity}")

# Decision based on similarity and perplexity
similarity_threshold = 0.5
perplexity_threshold = 50

if query_response_similarity < similarity_threshold or response_perplexity > perplexity_threshold:
    print("The response is likely hallucinated.")
else:
    print("The response is likely accurate.")
