import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# Wolfram Alpha
WOLFRAM_ALPHA_APPID = os.environ['WOLFRAM_ALPHA_APPID']

# Flask
SERVER_HOST = os.environ['SERVER_HOST']
SERVER_PORT = int(os.environ['SERVER_PORT'])
