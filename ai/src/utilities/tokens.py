from langchain.callbacks import get_openai_callback


def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total {cb.total_tokens} tokens.')

    return result
