from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper

from framework.agent import AgentTool


def get_wolfram_alpha_tool() -> AgentTool:
    """Remember to create environment variable named WOLFRAM_ALPHA_APPID with your Wolfram API app id"""

    wolfram_api = WolframAlphaAPIWrapper()
    wolfram_tool = AgentTool(
        function=wolfram_api.run,
        name='Wolfram Alpha',
        description='Useful only for numerical calculation. Not useful for proof work.'
    )

    return wolfram_tool
