from langchain.tools import BaseTool
from langchain.agents import load_tools
from langchain.schema.language_model import BaseLanguageModel


def get_google_search_tool(llm: BaseLanguageModel) -> BaseTool:
    tools = load_tools(['google-serper'], llm=llm)
    return tools[0]
