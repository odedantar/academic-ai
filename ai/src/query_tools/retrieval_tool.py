from typing import Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import BaseTool, Tool
from langchain.schema.language_model import BaseLanguageModel

from utilities.models import get_openai_llm
from query_tools.subquery_writer import get_subqueries
from query_tools.engine_chooser import choose_query_engine
from query_tools.wikipedia import get_wikipedia_query_tool
from query_tools.serper_api import get_google_search_tool


MAX_SUMMARY_LENGTH = 250  # Word count

summary_template = """Summarize the text below, put emphasis on data and facts. 
Write only the summary and nothing more. Try to keep the summary under {summary_length} words.

TEXT:
{text}

Begin!

SUMMARY:
"""


def get_summary_chain(llm: BaseLanguageModel) -> LLMChain:
    summary_prompt = PromptTemplate.from_template(template=summary_template)
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

    return summary_chain


def summarize(llm: BaseLanguageModel, text: str) -> str:
    summary_chain = get_summary_chain(llm=llm)
    return summary_chain(
        inputs={
            'summary_length': MAX_SUMMARY_LENGTH,
            'text': text
        }
    )['text']


def get_retrieval_tool() -> BaseTool:
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

            subquery_map = {}
            for sq in subqueries:
                subquery_map[sq] = choose_query_engine(llm=query_llm, query=sq, query_engines=engine_descriptions)

            result_map = {}
            for sq, engine in subquery_map.items():
                result_map[sq] = engines[engine]['tool'].invoke(sq)

            output = []
            for sq, result in result_map.items():
                output.append(f'QUERY: {sq}\nSOURCE: {subquery_map[sq]}\nRESULT: {result}')

            return summarize(llm=query_llm, text='\n'.join(output))

        except Exception as e:
            return """Failed to preform retrieval, might be caused by an error in one of the subqueries. Try again, 
            but if you get another failed retrieval it might need some time for the problem to be fixed."""

    return Tool(
        name="Retrieval tool",
        func=tool_wrapper,
        description="Useful for fact retrieval using natural language querying."
    )
