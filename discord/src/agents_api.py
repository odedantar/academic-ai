import json
import asyncio
import requests
import threading
from queue import Queue
from typing import Dict, Awaitable
from requests import Response
from jsonschema import validate, ValidationError

from config import AI_API_URL


REQUEST_TIMEOUT = 180  # In seconds

# Define the JSON schema
answer_schema = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
        }
    }, "required": ["answer"]
}


async def task_request(task: str):
    event_loop = asyncio.get_event_loop()

    url = AI_API_URL + '/discord/task'
    headers = {'Content-Type': 'application/json'}
    json_data = json.dumps({'task': task})

    print("Sending API request to: " + url)
    request = event_loop.run_in_executor(None, lambda: requests.post(
        url=url,
        headers=headers,
        data=json_data,
        timeout=REQUEST_TIMEOUT
    ))

    while not request.done():
        await asyncio.sleep(1)
    response = await request

    try:
        response_json = response.json()
    except json.JSONDecodeError:
        print('Invalid JSON')
    else:
        try:
            validate(response_json, answer_schema)
        except ValidationError as e:
            print('Invalid JSON schema:', e)
        else:
            return response_json


async def task_stream(task: str, stream_queue: Queue):
    event_loop = asyncio.get_event_loop()

    url = AI_API_URL + '/discord/task/stream'
    headers = {'Content-Type': 'application/json'}
    json_data = json.dumps({'task': task})

    print("Sending API request to: " + url)
    request = event_loop.run_in_executor(None, lambda: requests.post(
        url=url,
        headers=headers,
        data=json_data,
        stream=True
    ))

    while not request.done():
        await asyncio.sleep(1)
    stream = await request

    stream_thread = threading.Thread(target=streamer, args=[stream, stream_queue])
    stream_thread.start()


def streamer(stream: Response, queue: Queue):
    print("Stream thread: Stream has began")
    for chunk in stream.iter_content(chunk_size=1024):
        if chunk:
            print("Stream thread: Putting chunk in queue...")
            queue.put(chunk.decode('utf-8'))

    queue.put(None)
