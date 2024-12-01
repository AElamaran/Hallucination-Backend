from langchain import hub
from langchain.agents import AgentExecutor,create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from get_model import get_bedrock_model

tools =[TavilySearchResults(max_results=5)]
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/react")
llm = get_bedrock_model()
agent = create_react_agent(llm, tools, prompt)
# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({
    "input": "What is Langchain"
})



# def get_ReAct_prompting_response(prompt:str):
#
#     prompt_temp=
#     agent = agent_executor.agent.invoke(
#         {
#
#
#         }
#     )

