from langchain.docstore.wikipedia import Wikipedia
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain.agents import Tool, AgentType, AgentExecutor, initialize_agent
from langchain.agents.react.base import DocstoreExplorer


def get_wikipedia_query_tool() -> Tool:
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    wikipedia_tool = Tool(
        name='Wikipedia',
        func=wikipedia.run,
        description='Useful for when you need to query wikipedia'
    )

    return wikipedia_tool


def get_wikipedia_agent(llm, is_verbose=False, max_iterations=3) -> AgentExecutor:
    docstore = DocstoreExplorer(Wikipedia())
    tools = [
        Tool(
            name="Search",
            func=docstore.search,
            description='search wikipedia'
        ),
        Tool(
            name="Lookup",
            func=docstore.lookup,
            description='lookup a term in wikipedia'
        )
    ]

    agent = initialize_agent(
        agent=AgentType.REACT_DOCSTORE,
        tools=tools,
        llm=llm,
        verbose=is_verbose,
        max_iterations=max_iterations,
        handle_parsing_errors=True
    )

    return agent


def get_wikipedia_sub_agent(llm, is_verbose=False, max_iterations=3) -> Tool:
    wiki_agent = get_wikipedia_agent(
        llm=llm,
        is_verbose=is_verbose,
        max_iterations=max_iterations
    )

    sub_agent = Tool(
        name='Wikipedia agent',
        func=wiki_agent.run,
        description='Useful for when you need to search wikipedia'
    )

    return sub_agent
