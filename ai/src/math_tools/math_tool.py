from queue import Queue
from typing import Optional
from langchain.tools import BaseTool, Tool
from langchain.agents import initialize_agent, AgentType

from utilities.models import get_openai_llm, get_conversational_memory
from utilities.streamers import StreamHandler, CodeBlockStreamHandler
from math_tools.wolfram_alpha import get_wolfram_alpha_tool
from math_tools.question_writer import get_question_writer_tool
from math_tools.question_solver import get_question_solver_tool
from math_tools.proofreader import get_proofreader_tool


def get_math_tool(
        max_iter: Optional[int] = 6,
        stream_queue: Optional[Queue] = None,
        timeout: Optional[int] = 180,
) -> BaseTool:
    # Streamers
    agent_streamer = None if not stream_queue else [StreamHandler(queue=stream_queue)]
    latex_streamer = None if not stream_queue else [CodeBlockStreamHandler(queue=stream_queue, block_type='latex')]

    # Language Models
    agent_llm = get_openai_llm(
        temperature=0.05,
        model_name='gpt-4-1106-preview',
        streamers=agent_streamer
    )
    writer_llm = get_openai_llm(
        temperature=0.1,
        model_name='gpt-4-1106-preview'
    )
    solver_llm = get_openai_llm(
        temperature=0.1,
        model_name='gpt-4-1106-preview'
    )
    proofread_llm = get_openai_llm(
        temperature=0.05,
        model_name='gpt-4-1106-preview'
    )
    wrapper_llm = get_openai_llm(
        temperature=0.1,
        model_name='gpt-4-1106-preview'
    )
    latex_llm = get_openai_llm(
        temperature=0.0,
        model_name='gpt-4-1106-preview',
        streamers=latex_streamer
    )

    # Tools
    wolfram_tool = get_wolfram_alpha_tool()
    question_writer = get_question_writer_tool(
        llm=writer_llm,
        latex_llm=latex_llm,
        wrapper_llm=wrapper_llm
    )
    question_solver = get_question_solver_tool(
        llm=solver_llm,
        latex_llm=latex_llm,
        wrapper_llm=wrapper_llm
    )
    proofreader = get_proofreader_tool(
        llm=proofread_llm,
        latex_llm=latex_llm,
        wrapper_llm=wrapper_llm
    )

    tools = [wolfram_tool, question_writer, question_solver, proofreader]

    # Agent
    math_agent = initialize_agent(
        llm=agent_llm,
        tools=tools,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        # memory=get_conversational_memory(),
        verbose=True,
        max_iterations=max_iter,
        handle_parsing_errors=True,
        max_execution_time=timeout
    )

    def tool_wrapper(input: Optional[str] = None) -> str:
        if not input:
            return """Could not continue with an empty input"""

        try:
            return math_agent.invoke({'input': input})['output']

        except Exception as e:
            return """Failed to invoke math agent"""

    # Math tool
    return Tool(
        name="Math tool",
        func=tool_wrapper,
        description="Useful for doing anything from simple calculations to advanced math."
                    "Can write, solve, or proofread mathematical texts."
    )