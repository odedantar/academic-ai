from typing import Optional
from langchain.schema.language_model import BaseLanguageModel

from gemini_utils.models import get_gemini_llm
from framework.agent import Agent
from search_toolkit.tool import get_search_tool


clarifications = """Do not use the Search tool to ask for queries with implicit references to information from your 
workflow, here are a few example of bad queries:
    X Tool Input: What is the topic of the document? - The tool doesn't know what document you're referring to.
    X Tool Input: What are the key concepts mentioned in the abstract? - The tool doesn't know what abstract you're talking about.
"""


def get_writer_agent(llm: Optional[BaseLanguageModel] = None, max_iterations: Optional[int] = 10) -> Agent:

    llm = get_gemini_llm() if llm is None else llm
    tools = [get_search_tool(llm=llm)]

    return Agent(
        llm=llm,
        tools=tools,
        max_iterations=max_iterations,
        clarifications=clarifications
    )
