from queue import Queue

from openai_utils.models import get_openai_llm
from query_tools.retrieval_tool import get_retrieval_tool
from math_tools.math_tool import get_math_tool
from framework.agent import Agent
from framework.agent_stream import AgentStream

MAX_ITERATIONS = 10  # Num. of iterations
AGENT_TIMEOUT = 240  # In seconds


def build_agent(stream_queue: Queue = None):

    gpt3 = get_openai_llm(
        temperature=0.05,
        model_name='gpt-3.5-turb'
    )
    gpt4 = get_openai_llm(
        temperature=0.05,
        model_name='gpt-4-1106-preview'
    )

    queue = Queue() if not stream_queue else stream_queue
    stream = AgentStream(queue=queue, is_verbose=True)

    retrieval_tool = get_retrieval_tool(llm=gpt4)
    math_tool = get_math_tool(max_iter=MAX_ITERATIONS)

    # when giving tools to LLM, we must pass as list of tools
    tools = [retrieval_tool, math_tool]

    agent = Agent(llm=gpt4, tools=tools, max_iterations=MAX_ITERATIONS, stream=stream)
    return agent
