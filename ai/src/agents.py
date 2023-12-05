from queue import Queue
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from utilities.config import OPENAI_API_KEY
from utilities.stream import StreamHandler, CodeBlockStreamHandler
from search_tools.wikipedia import get_wikipedia_sub_agent
from math_tools.wolfram_alpha import get_wolfram_alpha_tool
from math_tools.question_writer import get_question_writer_tool
from math_tools.question_solver import get_question_solver_tool

MAX_ITERATIONS = 6  # Num. of iterations
AGENT_TIMEOUT = 180  # In seconds


def get_openai_llm(
        openai_key: str,
        temperature=0.0,
        model_name='gpt-3.5-turbo',
        callbacks: List = None
) -> ChatOpenAI:

    return ChatOpenAI(
        openai_api_key=openai_key,
        temperature=temperature,
        model_name=model_name,
        streaming=False if not callbacks else True,
        callbacks=callbacks
    )


def get_conversational_memory() -> ConversationBufferWindowMemory:

    return ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )


def get_math_agent(queue: Queue = None):
    gpt3 = get_openai_llm(
        openai_key=OPENAI_API_KEY,
        temperature=0.1,
        model_name='gpt-3.5-turbo'
    )
    gpt4 = get_openai_llm(
        openai_key=OPENAI_API_KEY,
        temperature=0.1,
        model_name='gpt-4-1106-preview'
    )

    agent_llm = get_openai_llm(
        openai_key=OPENAI_API_KEY,
        temperature=0.05,
        model_name='gpt-4-1106-preview',
        callbacks=None if not queue else [StreamHandler(queue=queue)]
    )
    tool_llm = get_openai_llm(
        openai_key=OPENAI_API_KEY,
        temperature=0.1,
        model_name='gpt-4-1106-preview',
        callbacks=None if not queue else [StreamHandler(queue=queue)]
    )
    latex_llm = get_openai_llm(
        openai_key=OPENAI_API_KEY,
        temperature=0.0,
        model_name='gpt-4-1106-preview',
        callbacks=None if not queue else [CodeBlockStreamHandler(queue=queue, block_type='latex')]
    )

    wiki_tool = get_wikipedia_sub_agent(llm=tool_llm, is_verbose=True)
    wolfram_tool = get_wolfram_alpha_tool()
    question_writer = get_question_writer_tool(
        llm=gpt4,
        latex_llm=latex_llm,
        wrapper_llm=gpt4
    )
    question_solver = get_question_solver_tool(
        llm=gpt4,
        proofread_llm=gpt4,
        latex_llm=latex_llm,
        wrapper_llm=gpt4
    )

    # when giving tools to LLM, we must pass as list of tools
    tools = [wiki_tool, wolfram_tool, question_writer, question_solver]

    return initialize_agent(
        llm=agent_llm,
        tools=tools,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=get_conversational_memory(),
        verbose=True,
        max_iterations=MAX_ITERATIONS,
        handle_parsing_errors=True,
        max_execution_time=AGENT_TIMEOUT
    )
