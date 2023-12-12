import asyncio
import threading
from queue import Queue
from flask import Flask, Response, request, jsonify, stream_with_context
from jsonschema import validate, ValidationError

from agent_builder import build_agent
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
        agent = build_agent()
        event_loop = asyncio.get_event_loop()
        
        result = await event_loop.run_in_executor(None, lambda: agent.invoke(task))
        return jsonify({"answer": result}), 200
        # return jsonify({"answer": result['output']}), 200

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
        agent = build_agent(stream_queue=queue)

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


if __name__ == '__main__':
    app.run(host=SERVER_HOST, port=SERVER_PORT)
