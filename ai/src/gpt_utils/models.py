from typing import List, Optional
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from config import OPENAI_API_KEY


def get_openai_llm(
        temperature: Optional[float] = 0.0,
        model_name: Optional[str] = 'gpt-3.5-turbo',
        streamers: Optional[List[BaseCallbackHandler]] = None
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