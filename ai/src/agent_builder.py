from queue import Queue
from langchain.agents import initialize_agent, AgentType

from utilities.models import get_openai_llm, get_conversational_memory
from utilities.streamers import StreamHandler
from query_tools.retrieval_tool import get_retrieval_tool
from math_tools.math_tool import get_math_tool
from agents.custom_agent import CustomAgent

MAX_ITERATIONS = 6  # Num. of iterations
AGENT_TIMEOUT = 240  # In seconds


def build_agent(stream_queue: Queue = None):
    agent_streamer = None if not stream_queue else [StreamHandler(queue=stream_queue)]

    agent_llm = get_openai_llm(
        temperature=0.05,
        model_name='gpt-4-1106-preview',
        streamers=agent_streamer
    )

    retrieval_tool = get_retrieval_tool()
    math_tool = get_math_tool(
        max_iter=MAX_ITERATIONS,
        stream_queue=stream_queue,
        timeout=120
    )

    # when giving tools to LLM, we must pass as list of tools
    tools = [retrieval_tool, math_tool]

    agent = CustomAgent(llm=agent_llm, tools=tools, max_iterations=MAX_ITERATIONS)
    return agent

    # return initialize_agent(
    #     llm=agent_llm,
    #     tools=tools,
    #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     memory=get_conversational_memory(),
    #     verbose=True,
    #     max_iterations=MAX_ITERATIONS,
    #     handle_parsing_errors=True,
    #     max_execution_time=AGENT_TIMEOUT
    # )