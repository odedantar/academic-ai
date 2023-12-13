from queue import Queue
from typing import Optional

from framework.agent_tool import AgentTool
from framework.agent import Agent
from framework.agent_stream import AgentStream
from openai_utils.models import get_openai_llm
from math_tools.wolfram_alpha import get_wolfram_alpha_tool
from math_tools.question_writer import get_question_writer_tool
from math_tools.question_solver import get_question_solver_tool
from math_tools.proofreader import get_proofreader_tool
from math_tools.latex_writer import get_latex_tool


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

    # Tools
    wolfram_tool = get_wolfram_alpha_tool()
    question_writer = get_question_writer_tool(
        llm=gpt4,
        latex_llm=gpt3,
        wrapper_llm=gpt4
    )
    question_solver = get_question_solver_tool(
        llm=gpt4,
        latex_llm=gpt3,
        wrapper_llm=gpt4
    )
    proofreader = get_proofreader_tool(
        llm=gpt4,
        latex_llm=gpt3,
        wrapper_llm=gpt4
    )
    latex_typer = get_latex_tool(
        llm=gpt3,
        wrapper_llm=gpt4
    )

    tools = [wolfram_tool, question_writer, question_solver, proofreader, latex_typer]

    # Streamer
    queue = Queue() if not stream_queue else stream_queue
    stream = AgentStream(queue=queue, is_verbose=True)

    # Agent
    math_agent = Agent(llm=gpt4, tools=tools, max_iterations=max_iter, stream=stream)

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
        description="Useful for anything from simple calculations to advanced math, writing, solving, "
                    "or proofreading mathematical texts and questions using LaTeX syntax."
    )
