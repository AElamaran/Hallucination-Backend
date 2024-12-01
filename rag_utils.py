from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import Chroma
from get_model import get_bedrock_model
from get_embedding import get_embedding_function
CHROMA_DB_PATH = "database"
from prompt import PROMPT_TEMPLATE
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