from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun

from framework.agent import AgentTool


def get_wikipedia_tool() -> AgentTool:
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    wikipedia_tool = AgentTool(
        function=wikipedia.run,
        name='Wikipedia',
        description='Useful for when you need to query wikipedia'
    )

    return wikipedia_tool
