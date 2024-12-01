import argparse
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import Chroma
from get_model import get_bedrock_model
from get_embedding import get_embedding_function
import numpy as np
from prompt import PROMPT_TEMPLATE, PROMPT_TEMPLATE_WITH_PROMPT_ENGINEERING, BASIC_PROMPT_TEMPLATE
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import joblib
from ReAact_propmt_service import decompose_query,generate_thoughts,fetch_answers_for_thoughts

CHROMA_DB_PATH = "database"
prediction = 0.25
# Load the trained model and scaler
classifier = joblib.load('best_rf_clf.pkl')
scaler = joblib.load('scaler.pkl')


# RAG Response
def response_rag_pipeline(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(question=query_text, context=context_text)
    model = get_bedrock_model()
    response_text = model.invoke(prompt)
    formatted_response = f"Response: {response_text}"

    # Calculate embeddings
    query_embedding = get_bedrock_embeddings(query_text)
    context_embedding = get_bedrock_embeddings(context_text)
    response_embedding = get_bedrock_embeddings(response_text)

    # Calculate similarities
    query_response_similarity = calculate_cosine_similarity(query_embedding, response_embedding)
    context_response_similarity = calculate_cosine_similarity(context_embedding, response_embedding)

    # Prepare features for the classifier
    features = np.array([[query_response_similarity, context_response_similarity, 0]])  # Perplexity is set to 0 since it's not used here
    features = scaler.transform(features)

    # Predict hallucination
    hallucination_score = classifier.predict_proba(features)[0][1]
    if (hallucination_score>prediction):
        topics=decompose_query(query_text,response_text,context_text)
        print("These are topics in Rag Services",topics)
        thoughts= generate_thoughts(query_text,topics)
        print("These are Thoughts", thoughts)
        act_answers=fetch_answers_for_thoughts(query_text,thoughts)
        return {
            'question': query_text,
            'response': response_text,
            'context': context_text,
            'response_type': "The Model Likely to be Hallucinate therefore Approaching RAG+ReAct Technique",
            'act_answers': act_answers
        }
    else:
        return {
            'question': query_text,
            'response': response_text,
            'response_type': "The Model's Response is Unlikely to be Hallucinate therefore Approaching RAG Approach"
        }



# Response from Foundation Model (AWS Bedrock)
def response_foundation_model(query_text: str):
    prompt_template = ChatPromptTemplate.from_template(BASIC_PROMPT_TEMPLATE)
    prompt = prompt_template.format(question=query_text)
    model = get_bedrock_model()
    response_text = model.invoke(prompt)
    formatted_response = f"Response: {response_text}"
    return {
        'question': query_text,
        'response': response_text,
    }


def get_bedrock_embeddings(text):
    embedding_function = get_embedding_function()
    embeddings = embedding_function.embed_documents([text])
    return np.array(embeddings).squeeze()


def calculate_cosine_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

def get_bedrock_embeddings(text):
    embedding_function = get_embedding_function()
    embeddings = embedding_function.embed_documents([text])
    return np.array(embeddings).squeeze()

def calculate_perplexity(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()

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

    analysis_report = {
        'question': query,
        'response': response,
        'query_response_similarity': query_response_similarity,
        'context_response_similarity': context_response_similarity,
        'response_perplexity': response_perplexity
    }

    return analysis_report


def pure_response_rag_pipeline(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(question=query_text, context=context_text)
    model = get_bedrock_model()
    response_text = model.invoke(prompt)
    return {
        'prompt': query_text,
        'response': response_text
    }

