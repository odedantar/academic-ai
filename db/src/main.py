from flask import Flask, request, jsonify
from jsonschema import validate, ValidationError

from config import VECTOR_STORE_PATH, SERVER_HOST, SERVER_PORT
from vector_store import CustomVectorStore


store = CustomVectorStore(path_to_index=VECTOR_STORE_PATH)
app = Flask(__name__)

vector_search_schema = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string"
        },
        "k": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10
        }
    }, "required": ["query"]
}


@app.route("/vector/search", methods=['POST'])
async def faiss_search():
    data = request.get_json()

    if data is None:
        return jsonify({"error": "Empty JSON"}), 400

    try:
        validate(request.json, vector_search_schema)

    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

    query = request.json["query"]
    k = request.json["k"]

    try:
        documents = await store.asearch(query=query, k=k)
        results = [{
            "id": i + 1,
            "data": documents[i].page_content,
            "metadata": documents[i].metadata
        } for i in range(k)]

        return jsonify({"results": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(host=SERVER_HOST, port=SERVER_PORT)

