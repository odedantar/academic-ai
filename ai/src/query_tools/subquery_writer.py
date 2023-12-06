import json
from typing import Optional, List
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import BaseTool, Tool
from langchain.schema.language_model import BaseLanguageModel

from query_tools.engine_chooser import choose_query_engine


subquery_template = """You are given a query:

QUERY: 
{query}

Break it down to sub queries which are optimized for vector similarity search. Break it down with as few sub queries 
as possible. When writing the sub queries follow the format of the JSON scheme below:

JSON:
{json_scheme}

Pay attention - Write only the JSON scheme and nothing more.

Begin!

JSON:
"""

json_scheme = """{
    "queries": [
        "1-st sub query",
        "2-nd sub query",
        ...
        "n-th sub query"
    ]
}"""


def get_subquery_chain(llm: BaseLanguageModel) -> LLMChain:
    subquery_prompt = PromptTemplate.from_template(subquery_template)
    subquery_chain = LLMChain(llm=llm, prompt=subquery_prompt)

    return subquery_chain


def get_subqueries(llm: BaseLanguageModel, query: str) -> List[str]:
    subquery_chain = get_subquery_chain(llm)
    response = subquery_chain(
        inputs={
            'query': query,
            'json_scheme': json_scheme
        })
    scheme = response['text']
    scheme = scheme.replace('```json', '')
    scheme = scheme.replace('```', '')
    try:
        scheme = json.loads(scheme, strict=False)
    except ValueError as e:
        raise e

    return scheme['queries']


# def get_subquery_tool(
#         llm: BaseLanguageModel,
#         tool_name: str,
#         tool_description: str) -> Tool:
#
#     def tool_wrapper(query: Optional[str] = None) -> str:
#         if not query:
#             return """Could not continue with an empty query"""
#
#         try:
#             sub_queries = get_subqueries(llm=llm, query=query)
#
#         except ValueError as e:
#             return """Failed at generating sub queries."""
#
#     return Tool(
#         name=tool_name,
#         func=tool_wrapper,
#         description=tool_description
#     )


