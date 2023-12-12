from multiprocessing import Process, Manager

from agent_builder import build_agent


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
    agent = build_agent()
    text = in_args['text']
    # out_args['answer'] = agent.invoke(text)['output']
    out_args['answer'] = agent.invoke(text)


if __name__ == '__main__':
    local_run()
