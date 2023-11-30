from flask import Flask, request
from multiprocessing import Process, Manager

from config import SERVER_HOST, SERVER_PORT
from agents import agent

app = Flask(__name__)


@app.route("/math", methods=['GET', 'POST'])
def main():
    try:
        print(request.values['question'])
        return agent.invoke(request.values['question'])['output']

    except Exception as e:
        print(e)
        return """AI agent could not parse the request."""


def local():
    def worker(in_args: dict, out_args: dict):
        question = in_args['question']
        out_args['answer'] = agent.invoke(question)['output']

    manager = Manager()
    timeout = 180  # Seconds

    print(request.values['question'])

    in_args = {'question': request.values['question']}
    out_args = manager.dict()  # This is the dict we can access both in and out of the processes.

    proc = Process(target=worker, args=(in_args, out_args))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        return """The AI agent took too long.\nThe request timed out."""
    else:
        return out_args['answer']


if __name__ == '__main__':
    app.run(host=SERVER_HOST, port=SERVER_PORT)
