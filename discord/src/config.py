import os
from dotenv import load_dotenv


load_dotenv()

# Discord
DISCORD_TOKEN = os.environ['DISCORD_TOKEN']
GUILD_ID = int(os.environ['GUILD_ID'])

# API
API_BASE_URL = os.environ['API_BASE_URL']
