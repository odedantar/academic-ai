import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

WOLFRAM_ALPHA_APPID = os.environ['WOLFRAM_ALPHA_APPID']

DISCORD_TOKEN = os.environ['DISCORD_TOKEN']
GUILD_ID = int(os.environ['GUILD_ID'])