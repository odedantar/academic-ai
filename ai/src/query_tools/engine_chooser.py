import json
from typing import Dict
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.language_model import BaseLanguageModel


engine_template = """You are given a query and a list of query engine names with their description:

QUERY: 
{query}

ENGINES:
{engines}

Decide which engine is best for the query you were given. 
Write the name of the engine you chose in the format of the JSON scheme below:

JSON:
{json_scheme}

Pay attention - Write only the JSON scheme and nothing more.

Begin!

JSON:
"""

json_scheme = """{"engine": "engine name"}"""


def get_chooser_chain(llm: BaseLanguageModel) -> LLMChain:
    engine_prompt = PromptTemplate.from_template(engine_template)
    engine_chain = LLMChain(llm=llm, prompt=engine_prompt)

    return engine_chain


async def achoose_engine(llm: BaseLanguageModel, query: str, query_engines: Dict[str, str]) -> str:
    chooser_chain = get_chooser_chain(llm=llm)
    engines = ',\n\t'.join([f'"{name}": "{desc}"' for name, desc in query_engines.items()])
    engines = '{\n\t' + engines + '\n}'

    response = await chooser_chain.ainvoke(
        input={
            'query': query,
            'engines': engines,
            'json_scheme': json_scheme
        })
    scheme = response['text']
    scheme = scheme.replace('```json', '')
    scheme = scheme.replace('```', '')

    try:
        scheme = json.loads(scheme, strict=False)
    except ValueError as e:
        raise e

    return scheme['engine']


def choose_engine(llm: BaseLanguageModel, query: str, query_engines: Dict[str, str]) -> str:
    chooser_chain = get_chooser_chain(llm=llm)
    engines = ',\n\t'.join([f'"{name}": "{desc}"' for name, desc in query_engines.items()])
    engines = '{\n\t' + engines + '\n}'

    response = chooser_chain.invoke(
        inputs={
            'query': query,
            'engines': engines,
            'json_scheme': json_scheme
        })
    scheme = response['text']
    scheme = scheme.replace('```json', '')
    scheme = scheme.replace('```', '')

    try:
        scheme = json.loads(scheme, strict=False)
    except ValueError as e:
        raise e

    return scheme['engine']
