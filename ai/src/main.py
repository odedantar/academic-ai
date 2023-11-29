from agent import agent
from multiprocessing import Process, Manager


def main():
    manager = Manager()
    timeout = 180  # Seconds

    question = """No need for anything really, just wanted to say hello."""

    in_args = {'question': question}
    out_args = manager.dict()  # This is the dict we can access both in and out of the processes.

    proc = Process(target=worker, args=(in_args, out_args))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
    else:
        print(out_args['answer'])


def worker(in_args: dict, out_args: dict):
    question = in_args['question']
    out_args['answer'] = agent.invoke(question)['output']


if __name__ == '__main__':
    main()
