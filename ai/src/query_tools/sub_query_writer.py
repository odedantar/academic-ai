import json
from typing import List
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.language_model import BaseLanguageModel


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


def get_sub_query_chain(llm: BaseLanguageModel) -> LLMChain:
    subquery_prompt = PromptTemplate.from_template(subquery_template)
    subquery_chain = LLMChain(llm=llm, prompt=subquery_prompt)

    return subquery_chain


def get_sub_queries(llm: BaseLanguageModel, query: str) -> List[str]:
    sub_query_chain = get_sub_query_chain(llm)
    response = sub_query_chain(
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
