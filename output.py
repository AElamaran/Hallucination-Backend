from prompt import FINAL_PROMPT_TEMPLETE
from get_model import get_bedrock_model
from langchain.prompts import ChatPromptTemplate

def generate_final_prompt(response_data: dict):
    # Access the nested "response" key from the provided response_data
    response_content = response_data

    # Extract components
    query = response_content.get('question', '')
    context = response_content.get('context', '')
    response = response_content.get('response', '')
    act_answers = response_content.get('act_answers', {})
    response_type=response_content.get('response_type','')

    # Format act_answers into a readable string
    act_answers_str = ""
    for thought_key, content in act_answers.items():
        thought = content.get('thought', '')
        answer = content.get('answer', '')
        act_answers_str += f"{thought_key}: {thought}\nAnswer: {answer}\n\n"

    prompt_templete=ChatPromptTemplate.from_template(FINAL_PROMPT_TEMPLETE)
    final_prompt = prompt_templete.format(
        query=query,
        context=context,
        response=response,
        act_answers=act_answers_str,
        response_type= response_type
    )
    model= get_bedrock_model()
    final_answer= model.invoke(final_prompt)

    return {
        'question':query,
        'final_response': final_answer,
        'approach':response_type
    }







