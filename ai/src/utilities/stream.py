"""
Reference to the original idea for the streaming queue:
https://stackoverflow.com/questions/76284412/stream-a-response-from-langchains-openai-with-pyton-flask-api
"""


from queue import Queue
from typing import Any, Dict, List
from langchain.schema import LLMResult
from langchain.callbacks.base import BaseCallbackHandler


class StreamHandler(BaseCallbackHandler):
    """Useful documentation: https://python.langchain.com/docs/modules/callbacks/"""
    def __init__(self, queue: Queue):
        self.content = ""
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.content += token.lower()
        self.queue.put(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        if 'final answer:' in self.content:
            self.queue.put(None)
        else:
            self.content = ""
            self.queue.put('\n')


class CodeBlockStreamHandler(BaseCallbackHandler):
    """Useful documentation: https://python.langchain.com/docs/modules/callbacks/"""
    def __init__(self, queue: Queue, block_type: str = ''):
        self.content = ""
        self.queue = queue
        self.type = block_type

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        self.queue.put('\n**Tool:**\n')
        self.queue.put('\n```' + self.type + '\n')

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        t = token.strip()
        if t != '```' + self.type and t != '```' and t != self.type:
            self.content += token.lower()
            self.queue.put(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        self.content = ""
        self.queue.put('\n```\n')


class SilentPhaseStreamHandler(BaseCallbackHandler):
    """Useful documentation: https://python.langchain.com/docs/modules/callbacks/"""
    def __init__(self, queue: Queue, phase_name: str):
        self.content = ""
        self.queue = queue
        self.name = phase_name

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        self.queue.put('\n**Starting:** {name}\n'.format(name=self.name))

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.content += token.lower()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        self.content = ""
        self.queue.put('\n**Parsing:** {name}\n'.format(name=self.name))
