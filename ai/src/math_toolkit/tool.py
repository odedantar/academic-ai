from queue import Queue
from typing import Optional

from gpt_utils.models import get_openai_llm
from gemini_utils.models import get_gemini_llm
from framework.agent import Agent, AgentTool, AgentStream
from math_toolkit.wolfram_alpha import get_wolfram_alpha_tool
from math_toolkit.question_writer import get_question_writer_tool
from math_toolkit.question_solver import get_question_solver_tool
from math_toolkit.proofreader import get_proofreader_tool
from math_toolkit.latex_writer import get_latex_tool


def get_math_tool(
        max_iter: Optional[int] = 6,
        stream_queue: Optional[Queue] = None,
        timeout: Optional[int] = 180
) -> AgentTool:

    # Language Models
    gpt3 = get_openai_llm(
        temperature=0.05,
        model_name='gpt-3.5-turbo'
    )
    gpt4 = get_openai_llm(
        temperature=0.05,
        model_name='gpt-4-1106-preview'
    )
    gemini_pro = get_gemini_llm(
        model_name='gemini-pro'
    )

    # Tools
    wolfram_tool = get_wolfram_alpha_tool()
    question_writer = get_question_writer_tool(
        llm=gemini_pro,
        latex_llm=gemini_pro,
        wrapper_llm=gemini_pro
    )
    question_solver = get_question_solver_tool(
        llm=gemini_pro,
        latex_llm=gemini_pro,
        wrapper_llm=gemini_pro
    )
    proofreader = get_proofreader_tool(
        llm=gemini_pro,
        latex_llm=gemini_pro,
        wrapper_llm=gemini_pro
    )
    latex_typer = get_latex_tool(
        llm=gemini_pro,
        wrapper_llm=gemini_pro
    )

    tools = [wolfram_tool, question_writer, question_solver, proofreader, latex_typer]

    # Streamer
    queue = Queue() if not stream_queue else stream_queue
    stream = AgentStream(queue=queue, is_verbose=True)

    # Agent
    math_agent = Agent(
        llm=gemini_pro,
        tools=tools,
        max_iterations=max_iter,
        stream=stream
    )

    async def wrapper(input: Optional[str] = None) -> str:
        if not input:
            return """Could not continue with an empty input"""

        try:
            return await math_agent.invoke(input)

        except Exception as e:
            print(e)
            return """Failed to invoke math agent"""

    # Math tool
    return AgentTool(
        function=wrapper,
        name="Mathematical toolkit",
        description="Useful for both numerical calculations and advanced math. "
                    "Writing, solving, or proofreading mathematical texts and problems in LaTeX syntax."
    )


if __name__ == '__main__':
    import asyncio
    from framework.research_agent import ResearchAgent
    from search_toolkit.tool import get_search_tool

    gemini_pro = get_gemini_llm(
        model_name='gemini-pro'
    )

    wolfram_tool = get_wolfram_alpha_tool()
    wolfram_tool.name = "Calculator"
    tools = [get_search_tool(gemini_pro), wolfram_tool]

    agent = ResearchAgent(
        llm=gemini_pro,
        tools=tools,
        max_iterations=20
    )

    question = """Give me the list of definitions and theorems I need to know to excell at Calculus I"""

    loop = asyncio.get_event_loop()
    loop.run_until_complete(agent.invoke(question))
