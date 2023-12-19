import asyncio
from typing import Optional
from langchain.schema.language_model import BaseLanguageModel

from documents_toolkit.writing_utils import get_writer_agent


def write_abstract(description: str, llm: Optional[BaseLanguageModel] = None) -> str:
    request = f'Write an abstract for a document based on this description:\n"{description}"'
    loop = asyncio.get_running_loop()
    writer = get_writer_agent(llm)

    return loop.run_until_complete(writer.invoke(request))


async def awrite_abstract(description: str, llm: Optional[BaseLanguageModel] = None) -> str:
    request = f'Write an abstract for a document based on this description:\n"{description}"\n' \
              f'When writing the answer write only the document\'s abstract and nothing else.'
    writer = get_writer_agent(llm)

    return await writer.invoke(request)
