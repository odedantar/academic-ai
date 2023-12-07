import threading
from queue import Queue
from flask import Flask, Response, request, stream_with_context
from multiprocessing import Process, Manager

from agent_builder import build_agent
from utilities.config import SERVER_HOST, SERVER_PORT

STREAM_CHUNK_SIZE = 50

app = Flask(__name__)


@app.route("/math", methods=['GET', 'POST'])
def math():
    try:
        agent = build_agent()
        return agent.invoke(request.values['text'])['output']

    except Exception as e:
        print(e)
        return """AI agent could not parse the request."""


@app.route("/math/stream", methods=['GET', 'POST'])
def math_stream():
    def streamer(q: Queue):
        token = ""
        while token is not None:
            chunk = ""
            for i in range(STREAM_CHUNK_SIZE):
                token = q.get()
                if not token:
                    break
                else:
                    chunk += token
            yield chunk

    try:
        queue = Queue()
        stream = streamer(queue)

        agent = build_agent(stream_queue=queue)
        invoke = threading.Thread(target=agent.invoke, args=[request.values['text']])
        invoke.start()

        return Response(stream_with_context(stream))

    except Exception as e:
        print(e)
        return """AI agent could not parse the request."""


def local_run():
    manager = Manager()
    timeout = 240  # Seconds

    text = """Who lives longer, dogs, cats or wolves?"""

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
    agent = build_agent()
    text = in_args['text']
    # out_args['answer'] = agent.invoke(text)['output']
    out_args['answer'] = agent.run(text)


if __name__ == '__main__':
    # app.run(host=SERVER_HOST, port=SERVER_PORT)
    local_run()
