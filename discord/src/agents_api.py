import asyncio
from requests import post
from config import API_BASE_URL

REQUEST_TIMEOUT = 180  # In seconds


async def math_bot(text: str) -> str:
    loop = asyncio.get_event_loop()

    url = API_BASE_URL + '/math'
    data = {'text': text}

    print("Sending API request to: " + url + "...")
    request = loop.run_in_executor(None, lambda: post(url=url, data=data, timeout=REQUEST_TIMEOUT))
    response = await request

    answer = response.content.decode('latin-1')
    print("Answer: '" + answer + "'")
    return answer


async def math_bot_stream(text: str):
    loop = asyncio.get_event_loop()

    url = API_BASE_URL + '/math/stream'
    data = {'text': text}

    print("Sending API request to: " + url + "...")
    request = loop.run_in_executor(None, lambda: post(url=url, data=data, stream=True))
    response = await request

    def stream():
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                yield chunk.decode('utf-8')

    return stream()


if __name__ == "__main__":
    resp = asyncio.run(math_bot("Hello"))
    print(resp)
