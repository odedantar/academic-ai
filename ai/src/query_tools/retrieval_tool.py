import asyncio
from typing import Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.language_model import BaseLanguageModel

from openai_utils.models import get_openai_llm
from framework.agent_tool import AgentTool
from query_tools.sub_query_writer import get_sub_queries
from query_tools.engine_chooser import achoose_engine
from query_tools.wikipedia import get_wikipedia_tool
from query_tools.serper_api import get_google_search_tool
from query_tools.vector_store import get_vector_store_tool


MAX_SUMMARY_LENGTH = 150  # Word count

summary_template = f"""You are given a main query and the results of several sub queries which were 
derived from the main one:

MAIN QUERY: 
{{query}}

RESULTS:
{{results}}

Summarize the results with the aim of answering the main query. Write only the summary and nothing more. 
Summarize based only on the results you were given. The summary should be no more than {MAX_SUMMARY_LENGTH} words.

Begin!

SUMMARY:
"""


def get_summary_chain(llm: BaseLanguageModel) -> LLMChain:
    summary_prompt = PromptTemplate.from_template(template=summary_template)
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

    return summary_chain


def summarize(llm: BaseLanguageModel, main_query: str, query_results: str) -> str:
    summary_chain = get_summary_chain(llm=llm)
    return summary_chain(
        inputs={
            'query': main_query,
            'results': query_results
        }
    )['text']


def get_retrieval_tool(llm: BaseLanguageModel) -> AgentTool:
    tools = [
        get_wikipedia_tool(),
        get_google_search_tool(llm=llm),
        get_vector_store_tool()
    ]

    engines = {
        t.name: {
            'description': t.description,
            'tool': t
        } for t in tools
    }
    engines_desc = {t.name: t.description for t in tools}

    async def wrapper(query: Optional[str] = None) -> str:
        if not query:
            return """Could not continue with an empty query"""

        try:
            sub_queries = get_sub_queries(llm=llm, query=query)

            # Step 1: Choose engines for each subquery in parallel
            choosers = [achoose_engine(llm=llm, query=sq, query_engines=engines_desc) for sq in sub_queries]
            engines_for_sub_queries = await asyncio.gather(*choosers)
            sub_query_map = dict(zip(sub_queries, engines_for_sub_queries))

            # Step 2: Perform sub queries in parallel
            invokes = [engines[engine]['tool'].invoke(sq) for sq, engine in sub_query_map.items()]
            results = await asyncio.gather(*invokes)
            result_map = dict(zip(sub_queries, results))

            output = []
            for sq, result in result_map.items():
                output.append(f'SUB QUERY: {sq}\nSOURCE: {sub_query_map[sq]}\nRESULT: {result}')

            return summarize(llm=llm, main_query=query, query_results='\n'.join(output))

        except Exception as e:
            return "Failed to preform retrieval, might be caused by an error in one of the sub queries. Try again," \
                   "but if you get another failed retrieval it might need some time for the problem to be fixed."

    return AgentTool(
        function=wrapper,
        name="Retrieval tool",
        description="Useful for information retrieval from the web using natural language querying."
    )
