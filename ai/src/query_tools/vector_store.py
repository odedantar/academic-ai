import json
import asyncio
import requests
from typing import List, Dict, Optional
from jsonschema import validate, ValidationError

from config import VS_API_URL
from framework.agent_tool import AgentTool


REQUEST_TIMEOUT = 180  # In seconds

# Define the JSON schema
results_schema = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "data": {"type": "string"},
                    "id": {"type": "integer"},
                    "metadata": {}
                }
            }, "required": ["data", "id", "metadata"]
        }
    }, "required": ["results"]
}


async def vector_search(query: str, k: Optional[int] = 3) -> List[Dict[str, str]]:
    url = VS_API_URL + '/vector/search'
    data = {'query': query, 'k': k}

    # Set the Content-Type header to application/json
    headers = {'Content-Type': 'application/json'}

    # Convert the data to a JSON string
    json_data = json.dumps(data)

    # Send the request with the JSON data
    event_loop = asyncio.get_running_loop()
    request = event_loop.run_in_executor(None, lambda: requests.post(url=url, headers=headers, data=json_data))
    while not request.done():
        await asyncio.sleep(1)
    response = await request

    # Check if the response is valid JSON
    try:
        response_json = response.json()
    except json.JSONDecodeError:
        print('Invalid JSON')
    else:
        # Validate the response against the schema
        try:
            validate(response_json, results_schema)
        except ValidationError as e:
            print('Invalid JSON schema:', e)
        else:
            return response_json['results']


def get_vector_store_tool() -> AgentTool:

    async def wrapper(query: str) -> str:
        results = await vector_search(query=query)
        return '\n\n'.join([result['data'] for result in results])

    return AgentTool(
        function=wrapper,
        name='Academic library',
        description='Useful for querying data from academic books and syllabi'
    )


if __name__ == "__main__":
    resp = asyncio.run(vector_search(query="Momentum", k=3))
    print(resp)
