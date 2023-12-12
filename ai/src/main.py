import asyncio
import threading
from queue import Queue
from flask import Flask, Response, request, jsonify, stream_with_context
from jsonschema import validate, ValidationError
from multiprocessing import Process, Manager

from agents import get_agent
from utilities.config import SERVER_HOST, SERVER_PORT


STREAM_CHUNK_SIZE = 50

app = Flask(__name__)

task_schema = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string"
        }
    }, "required": ["task"]
}


@app.route("/discord/task", methods=['POST'])
async def task():
    data = request.get_json()

    if data is None:
        return jsonify({"error": "Empty JSON"}), 400

    try:
        validate(request.json, task_schema)

    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

    task = request.json['task']

    try:
        agent = get_agent()
        event_loop = asyncio.get_event_loop()
        result = await event_loop.run_in_executor(None, lambda: agent.invoke(task))

        return jsonify({"answer": result['output']}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/discord/task/stream", methods=['POST'])
def task_stream():
    data = request.get_json()

    if data is None:
        return jsonify({"error": "Empty JSON"}), 400

    try:
        validate(request.json, task_schema)

    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

    task = request.json['task']

    try:
        queue = Queue()
        agent = get_agent(stream_queue=queue)

        thread = threading.Thread(target=agent.invoke, args=[task])
        thread.start()

        stream = streamer(queue)
        return Response(stream_with_context(stream))

    except Exception as e:
        return jsonify({"error": str(e)}), 400


def streamer(q: Queue):
    token = ""
    while token is not None:
        chunk = ""
        for i in range(STREAM_CHUNK_SIZE):
            token = q.get()
            if token is None:
                break
            else:
                chunk += token
        yield chunk


def local_run():
    manager = Manager()
    timeout = 240  # Seconds

    text = """What are the main topics in classical physics academic syllabus?"""

    in_args = {'text': text}
    out_args = manager.dict()  # This is the dict we can access both in and out of the processes.
    out_args['answer'] = ''

    proc = Process(target=worker, args=(in_args, out_args))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        return """The AI agent took too long.\nThe request timed out."""
    else:
        return out_args['answer']


def worker(in_args: dict, out_args: dict):
    agent = get_agent()
    text = in_args['text']
    out_args['answer'] = agent.invoke(text)['output']


if __name__ == '__main__':
    app.run(host=SERVER_HOST, port=SERVER_PORT)
    # local_run()
