from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from ai.src.config import OPENAI_API_KEY
from ai.src.retrievers.gmail import get_gmail_tool
from ai.src.retrievers.wikipedia import get_wikipedia_sub_agent
from ai.src.math.wolfram_alpha import get_wolfram_alpha_tool
from ai.src.math.question_writer import get_question_writer_tool
from ai.src.math.question_solver import get_question_solver_tool


def get_openai_llm(openai_key: str, temperature=0.0, model_name='gpt-3.5-turbo') -> ChatOpenAI:
    model = ChatOpenAI(
        openai_api_key=openai_key,
        temperature=temperature,
        model_name=model_name
    )

    return model


def get_conversational_memory() -> ConversationBufferWindowMemory:
    memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )

    return memory


gpt3 = get_openai_llm(openai_key=OPENAI_API_KEY, temperature=0.05, model_name='gpt-3.5-turbo')
gpt4 = get_openai_llm(openai_key=OPENAI_API_KEY, temperature=0.05, model_name='gpt-4-1106-preview')

gmail_tool = get_gmail_tool(llm=gpt3, is_verbose=True)
wiki_tool = get_wikipedia_sub_agent(llm=gpt3, is_verbose=True)

question_writer = get_question_writer_tool(llm=gpt3)
question_solver = get_question_solver_tool(llm=gpt3)
wolfram_tool = get_wolfram_alpha_tool()

# when giving tools to LLM, we must pass as list of tools
tools = [wiki_tool, wolfram_tool, question_writer, question_solver]

agent = initialize_agent(
    llm=gpt4,
    tools=tools,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=get_conversational_memory(),
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True,
    max_execution_time=60
)
