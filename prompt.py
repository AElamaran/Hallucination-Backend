PROMPT_TEMPLATE = '''
{context}

---
Answer the following question within the above context don't go exceed the context, 
question : {question}
'''

PROMPT_TEMPLATE_WITH_PROMPT_ENGINEERING = '''
I will provide you with a context enclosed in triple backticks (). Your task is to analyze the information within this context and answer questions exclusively based on the provided details.

Guidelines:

1. Strict Adherence: Base your responses solely on the information presented within the context. Do not introduce external knowledge or assumptions.
2. Conciseness and Relevance: Craft answers that are brief, factual, and directly pertinent to the questions asked.
3. Information Gaps: If a question cannot be answered due to insufficient information in the context, respond with: "The context does not provide this information."

{context}
```

With Respect to the Instructions given, Answer the following question : {question}
'''

BASIC_PROMPT_TEMPLATE = "Act as a Cricket Enthusiast and answer the following question. The question is {question} use your own knowledge Think like you are fantacy person with some sort of commensense Use MCC rule book for Cricket Rules related questions "

REACT_DECOMPOSE_QUERY_PROMPT_TEMPLATE = """
Question: {query}
Answer: {response}
Context: {context}

The provided answer contains hallucinations. To resolve this using the ReAct prompting technique, I need relevant topics about the question from the knowledge-base folder. These topics should be converted into thoughts.

Please provide the most suitable minimum of 2 and maximum of 5 thoughts related to the question in the following JSON format:
```json
[
  "thought1": "thought1",
  "thought2": "thought2",
  "thought3": "thought3",
  "thought4": "thought4",
  "thought5": "thought5"
]

Ensure each thought is distinct and directly related to the question.
Provide only the JSON output, without any additional text.
"""
prediction=0.5

FINAL_PROMPT_TEMPLETE = """
The following is a detailed breakdown of a query related to cricket rules. Please consider all provided information carefully and generate a coherent and accurate final answer.

### User Query:
{query}

### Context:
The context provided for this query includes relevant cricket rules and considerations:
{context}

### Initial Response:
The initial response generated by the RAG system is as follows:
{response}

### Thought and Act Answers:
The following thoughts and corresponding answers have been generated using the ReAct prompting technique to refine and provide clarity on specific aspects of the query:
{act_answers}

### Final Task:
Using the above information, provide a final answer to the user query. Ensure that the answer considers all details and is consistent with the cricket rules. Avoid any hallucination or unsupported assumptions.
"""
