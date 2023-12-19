from langchain.agents import load_tools
from langchain.schema.language_model import BaseLanguageModel

from framework.agent import AgentTool


def get_google_search_tool(llm: BaseLanguageModel) -> AgentTool:
    tools = load_tools(['google-serper'], llm=llm)
    tool = tools[0]

    return AgentTool(function=tool.invoke, name=tool.name, description=tool.description)
