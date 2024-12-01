from get_model import get_bedrock_model
from langchain.prompts import ChatPromptTemplate
from rag_utils import pure_response_rag_pipeline
from prompt import REACT_DECOMPOSE_QUERY_PROMPT_TEMPLATE


def decompose_query(query: str, response: str, context: str):
    model = get_bedrock_model()
    prompt_template = ChatPromptTemplate.from_template(REACT_DECOMPOSE_QUERY_PROMPT_TEMPLATE)
    prompt = prompt_template.format(query=query, response=response, context=context)
    topic_response = model.invoke(prompt)
    print("These are Topic Responses: ", topic_response)

    # Manually parse the topic response string to extract topics and details
    topics = []
    try:
        # Extract lines between the brackets
        start_index = topic_response.find("[") + 1
        end_index = topic_response.rfind("]")
        topic_lines = topic_response[start_index:end_index].strip().split('\n')

        for line in topic_lines:
            if line.strip():  # Skip empty lines
                key_value = line.split(": ")
                if len(key_value) == 2:
                    topic_key = key_value[0].strip().strip('"')
                    topic_value = key_value[1].strip().strip('",')
                    topics.append((topic_key, topic_value))

        print("Parsed Topics: ", topics)

    except Exception as e:
        print(f"Error parsing topics: {e}")

    return topics


def generate_thoughts(query: str, topics: list):
    template = "To answer the query '{query}', I need to {action} {topic}."

    actions = [
        "understand how",
        "identify",
        "examine",
        "consider",
        "review"
    ]

    thoughts = {}
    for i, (action, (topic_key, topic_value)) in enumerate(zip(actions, topics)):
        thoughts[f"thought{i + 1}"] = template.format(query=query, action=action, topic=topic_value)

    return thoughts


def fetch_answers_for_thoughts(query: str, thoughts: dict):
    """
    This function fetches answers for each thought using the RAG pipeline.
    """
    answers = {}
    for thought_key, thought_text in thoughts.items():
        # Extract the specific topic for clarity
        topic_query = thought_text.split("I need to")[1].strip('.').strip()
        print(f"Querying RAG for: {topic_query}")

        # Use the RAG pipeline to get a response for the specific thought topic
        rag_response = pure_response_rag_pipeline(topic_query)
        answer = rag_response['response'].strip()

        # Compile the thought and its respective answer
        answers[thought_key] = {
            'thought': thought_text,
            'answer': answer
        }

    return answers


