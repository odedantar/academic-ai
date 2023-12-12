import os
from dotenv import load_dotenv


load_dotenv()

# Discord
DISCORD_TOKEN = os.environ['DISCORD_TOKEN']
GUILD_ID = int(os.environ['GUILD_ID'])

# AI API
AI_API_URL = os.environ['AI_API_URL']
