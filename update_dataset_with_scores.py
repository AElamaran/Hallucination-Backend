import pandas as pd
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from get_model import get_bedrock_model
from get_embedding import get_embedding_function
from sklearn.metrics.pairwise import cosine_similarity
from prompt import PROMPT_TEMPLATE

# Load your dataset
df = pd.read_csv('dataset_h.csv', header=None, names=['Question', 'label'])

CHROMA_DB_PATH = "database"


def get_bedrock_embeddings(text):
    embedding_function = get_embedding_function()
    embeddings = embedding_function.embed_documents([text])
    return np.array(embeddings).squeeze()


def calculate_cosine_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]


def calculate_perplexity(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()


def response_rag_pipeline(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(question=query_text, context=context_text)
    model = get_bedrock_model()
    response_text = model.invoke(prompt)
    return {
        'question': query_text,
        'response': response_text,
        'context': context_text,
    }


def analyze_semantic_entropy(query_text: str):
    # Generate RAG response
    rag_result = response_rag_pipeline(query_text)
    query = rag_result['question']
    context = rag_result['context']
    response = rag_result['response']

    # Get embeddings using AWS Bedrock
    query_embedding = get_bedrock_embeddings(query)
    context_embedding = get_bedrock_embeddings(context)
    response_embedding = get_bedrock_embeddings(response)

    # Calculate similarities
    query_response_similarity = calculate_cosine_similarity(query_embedding, response_embedding)
    context_response_similarity = calculate_cosine_similarity(context_embedding, response_embedding)

    # Load a smaller model and tokenizer from Hugging Face for perplexity calculation
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    response_perplexity = calculate_perplexity(response, model, tokenizer)

    return {
        'query_response_similarity': query_response_similarity,
        'context_response_similarity': context_response_similarity,
        'response_perplexity': response_perplexity
    }


# Update DataFrame with new columns
df['context_response_similarity'] = np.nan
df['query_response_similarity'] = np.nan
df['response_perplexity'] = np.nan

# Iterate over each query and update DataFrame
for index, row in df.iterrows():
    query_text = row['Question']
    analysis_report = analyze_semantic_entropy(query_text)

    df.at[index, 'context_response_similarity'] = analysis_report['context_response_similarity']
    df.at[index, 'query_response_similarity'] = analysis_report['query_response_similarity']
    df.at[index, 'response_perplexity'] = analysis_report['response_perplexity']

# Save the updated DataFrame to a new CSV file
df.to_csv('updated_dataset.csv', index=False)

print("Dataset updated and saved as 'updated_dataset.csv'")
