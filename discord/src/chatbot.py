import copy
import discord
from discord import app_commands
from discord.ext import commands
from openai import AsyncOpenAI

from role import Role
from config import OPENAI_API_KEY, DISCORD_TOKEN, GUILD_ID


# Discord API data
intents = discord.Intents.all()
client = commands.Bot(command_prefix='!', intents=intents)
guild = discord.Object(id=GUILD_ID)


# Events
@client.event
async def on_ready():
    await client.tree.sync(guild=guild)
    print(f'{client.user} is now running!')


class Chatbot:
    def __init__(self, client):
        # Chat
        self.purpose = ''
        self.conversation = []

        # Discord
        self.client = client

    @staticmethod
    async def chat_completion(messages: list, model: str = "gpt-3.5-turbo") -> str:
        ai = AsyncOpenAI(api_key=OPENAI_API_KEY)

        completion = await ai.chat.completions.create(
            model=model,
            messages=messages
        )

        return completion.choices[0].message.content

    def add_message(self, role: Role, message: str) -> None:
        self.conversation.append({
            "role": role,
            "content": message
        })

    def clear_messages(self):
        self.conversation = []

    def add_purpose(self, messages: list) -> list:
        purposed = copy.deepcopy(messages)
        purposed.append({
            "role": Role.SYSTEM,
            "content": f'Your purpose: \n {self.purpose}'
        })

        return purposed

    async def get_response(self, message: str) -> str:
        self.add_message(Role.USER, message)

        response = await self.chat_completion(self.conversation)
        self.add_message(Role.ASSISTANT, response)

        return response

    def run(self) -> None:
        @client.tree.command(name='echo', guild=guild)
        @app_commands.describe(message='Echo back this message')
        async def echo(interaction: discord.Interaction, message: str):
            await interaction.response.send_message(message)

        @client.tree.command(name='chat', guild=guild)
        @app_commands.describe(message='Write your message here')
        async def chat(interaction: discord.Interaction, message: str):
            await interaction.response.defer()
            response = await self.get_response(message)
            await interaction.followup.send(response)

        @client.tree.command(name='clear', description='Clears bot\'s conversational history', guild=guild)
        @app_commands.describe()
        async def clear(interaction: discord.Interaction):
            self.clear_messages()
            await interaction.response.send_message('My chat history is cleared.')

        self.client.run(DISCORD_TOKEN)
