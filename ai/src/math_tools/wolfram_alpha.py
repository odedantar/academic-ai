from langchain.agents import Tool
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper


def get_wolfram_alpha_tool():
    """Remember to create environment variable named WOLFRAM_ALPHA_APPID with your Wolfram API app id"""

    wolfram_api = WolframAlphaAPIWrapper()
    wolfram_tool = Tool(
        name='Wolfram Alpha',
        func=wolfram_api.run,
        description='Useful for doing numerical calculation'
    )

    return wolfram_tool
