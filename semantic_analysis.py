import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Sample data
context = "Marie Curie conducted pioneering research on radioactivity."
query = "What did Marie Curie discover?"
response = "Marie Curie discovered radium and polonium."

# Vectorize texts for cosine similarity
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([query, response, context])
query_vector, response_vector, context_vector = vectors.toarray()

# Cosine Similarity
query_response_similarity = cosine_similarity([query_vector], [response_vector])[0][0]
context_response_similarity = cosine_similarity([context_vector], [response_vector])[0][0]

print(f"Cosine Similarity (Query vs. Response): {query_response_similarity}")
print(f"Cosine Similarity (Context vs. Response): {context_response_similarity}")

# BLEU Score
reference = [context.split()]  # List of reference sentences
candidate = response.split()
bleu_score = sentence_bleu(reference, candidate)
print(f"BLEU Score: {bleu_score}")

# ROUGE Score
rouge = Rouge()
rouge_scores = rouge.get_scores(response, context)
print(f"ROUGE Scores: {rouge_scores}")

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
