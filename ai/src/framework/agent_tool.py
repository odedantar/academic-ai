import inspect
import asyncio
from typing import Callable


class AgentTool:
    def __init__(self, function: Callable, name: str, description: str):
        self.function = function
        self.name = name
        self.description = description

    async def invoke(self, input: str) -> str:
        if inspect.iscoroutinefunction(self.function):
            return await self.function(input)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.function(input))
