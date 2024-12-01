import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def softmax(logits):
    """Compute softmax values for each set of logits."""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


def calculate_entropy(probabilities):
    """Calculate the entropy of a list of probabilities."""
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy


def calculate_token_probabilities(model, tokenizer, response):
    """Calculate the probabilities of each token in the response."""
    inputs = tokenizer(response, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    token_probs = probs.squeeze().numpy()
    return token_probs


def get_response_entropy(model, tokenizer, response):
    """Calculate the entropy for a response."""
    token_probs = calculate_token_probabilities(model, tokenizer, response)
    entropy = calculate_entropy(token_probs)
    return entropy


def generate_responses_with_entropy(model, tokenizer, prompt, num_responses=3):
    """Generate multiple responses and calculate their entropy."""
    inputs = tokenizer(prompt, return_tensors='pt')
    responses = []
    entropies = []
    for _ in range(num_responses):
        outputs = model.generate(inputs.input_ids, max_length=50, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(response)
        entropy = get_response_entropy(model, tokenizer, response)
        entropies.append(entropy)
    return responses, entropies


if __name__ == '__main__':
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "What is the capital of France?"
    responses, entropies = generate_responses_with_entropy(model, tokenizer, prompt)

    for i, (response, entropy) in enumerate(zip(responses, entropies), 1):
        print(f"Response {i}: {response}\nEntropy: {entropy}")
