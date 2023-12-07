from typing import Optional
from langchain.tools import BaseTool, Tool

from utilities.models import get_openai_llm
from query_tools.subquery_writer import get_subqueries
from query_tools.engine_chooser import choose_query_engine
from query_tools.wikipedia import get_wikipedia_query_tool
from query_tools.serper_api import get_google_search_tool


def get_query_tool() -> BaseTool:
    query_llm = get_openai_llm(
        temperature=0.25,
        model_name='gpt-4-1106-preview'
    )

    engines = {
        "wikipedia": {
            "description": "Largest online collaborative encyclopedia",
            "tool": get_wikipedia_query_tool()
        },
        "google": {
            "description": "The leading web search engine these days.",
            "tool": get_google_search_tool(llm=query_llm)
        }
        # "arxiv": {
        #     "description": "Largest academic papers archive in the fields of exact sciences",
        #     "tool": lambda: None
        # }
    }
    engine_descriptions = {name: engine['description'] for name, engine in engines.items()}

    def tool_wrapper(query: Optional[str] = None) -> str:
        if not query:
            return """Could not continue with an empty query"""

        try:
            subqueries = get_subqueries(llm=query_llm, query=query)

            subquery_map = {sq: choose_query_engine(llm=query_llm, query=sq, query_engines=engine_descriptions)
                            for sq in subqueries}

            result_map = {sq: engines[engine]['tool'].invoke(sq) for sq, engine in subquery_map.items()}

            return '\n\n'.join(
                [f'QUERY: {sq}\nSOURCE: {subquery_map[sq]}\nRESULT: {result}' for sq, result in result_map.items()]
            )

        except ValueError as e:
            return """Failed to preform query"""

    return Tool(
        name="Query tool",
        func=tool_wrapper,
        description="Useful for natural language querying and fact retrieval."
    )
