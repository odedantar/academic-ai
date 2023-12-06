from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from utilities.config import OPENAI_API_KEY


def get_openai_llm(
        temperature=0.0,
        model_name='gpt-3.5-turbo',
        streamers: List[BaseCallbackHandler] = None
) -> ChatOpenAI:

    return ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=temperature,
        model_name=model_name,
        streaming=False if not streamers else True,
        callbacks=streamers
    )


def get_conversational_memory() -> ConversationBufferWindowMemory:

    return ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )