import os
from dotenv import load_dotenv


load_dotenv()

# OpenAI
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# Wolfram Alpha
WOLFRAM_ALPHA_APPID = os.environ['WOLFRAM_ALPHA_APPID']

# Discord
DISCORD_TOKEN = os.environ['DISCORD_TOKEN']
GUILD_ID = int(os.environ['GUILD_ID'])

# Flask
SERVER_HOST = os.environ['SERVER_HOST']
SERVER_PORT = int(os.environ['SERVER_PORT'])
