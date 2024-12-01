from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

# Load model and tokenizer
model_name = "gpt2"  # Example model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_responses(model, tokenizer, query, num_responses=5):
    """Generate multiple candidate responses using the model."""
    inputs = tokenizer(query, return_tensors='pt')
    responses = []
    for _ in range(num_responses):
        outputs = model.generate(inputs.input_ids, max_length=50, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(response)
    return responses

def calculate_token_probabilities(model, tokenizer, response):
    """Calculate the probabilities of each token in the response."""
    inputs = tokenizer(response, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    token_probs = probs.squeeze().numpy()
    return token_probs

def calculate_entropy(probabilities):
    """Calculate the entropy of a list of probabilities."""
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy

def get_response_entropy(model, tokenizer, response):
    """Calculate the entropy for a response."""
    token_probs = calculate_token_probabilities(model, tokenizer, response)
    entropy = calculate_entropy(token_probs)
    return entropy

def get_response_with_low_entropy(model, tokenizer, query, threshold=1.0, num_responses=5):
    """Generate a response with low entropy."""
    responses = generate_responses(model, tokenizer, query, num_responses)
    best_response = None
    lowest_entropy = float('inf')

    for response in responses:
        entropy = get_response_entropy(model, tokenizer, response)

        if entropy < lowest_entropy:
            lowest_entropy = entropy
            best_response = response

    if lowest_entropy > threshold:
        print(lowest_entropy)
        return "High entropy detected. Please clarify your query."
    else:
        return best_response

# Example usage
query = "What is the capital of France?"
response = get_response_with_low_entropy(model, tokenizer, query, threshold=1.0)
print(response)
