import asyncio
from typing import Union, Optional
from queue import Queue

from agents import console


class CustomStream:
    def __init__(self, queue: Queue, is_verbose: Optional[bool] = False):
        self.q = queue
        self.is_verbose = is_verbose

    def write(self, input: Union[str, None]):
        if self.is_verbose and input is not None:
            console.highlight(text=input)

        self.q.put(input)

    async def awrite(self, input: Union[str, None]):
        raise NotImplementedError("CustomStream.awrite is not yet implemented")

    def read(self) -> Union[str, None]:
        if self.q.empty():
            return ""

        return self.q.get()

    async def aread(self) -> Union[str, None]:
        while self.q.empty():
            await asyncio.sleep(1)

        return self.q.get()



