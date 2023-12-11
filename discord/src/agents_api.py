import asyncio
import requests
import threading
from queue import Queue
from typing import Dict, Awaitable
from requests import Response

from config import AI_API_URL

REQUEST_TIMEOUT = 180  # In seconds


def math_bot(text: str) -> Awaitable[Response]:
    event_loop = asyncio.get_event_loop()

    url = AI_API_URL + '/math'
    data = {'text': text}

    print("Sending API request to: " + url + "...")
    request = event_loop.run_in_executor(None, lambda: requests.post(url=url, data=data, timeout=REQUEST_TIMEOUT))
    return request


def math_bot_stream(text: str, stream_queue: Queue):
    url = AI_API_URL + '/math/stream'
    data = {'text': text}

    def streamer(url: str, data: Dict, queue: Queue):
        print("Stream thread: Sending API request to: " + url + "...")
        stream = requests.post(url=url, data=data, stream=True)
        print("Stream thread: Stream has began")
        for chunk in stream.iter_content(chunk_size=1024):
            if chunk:
                print("Stream thread: Putting chunk in queue...")
                queue.put(chunk.decode('utf-8'))

        queue.put(None)

    stream_thread = threading.Thread(target=streamer, args=[url, data, stream_queue])
    stream_thread.start()


if __name__ == "__main__":
    resp = asyncio.run(math_bot("Hello"))
    print(resp)
