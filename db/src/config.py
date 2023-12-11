import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# Vector Store
VECTOR_STORE_PATH = os.environ['VECTOR_STORE_PATH']

# Flask
SERVER_HOST = os.environ['SERVER_HOST']
SERVER_PORT = int(os.environ['SERVER_PORT'])
