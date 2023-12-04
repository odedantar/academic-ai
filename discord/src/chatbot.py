import asyncio
import discord
from discord import app_commands
from discord.ext import commands

import agents_api as api
from config import DISCORD_TOKEN, GUILD_ID


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
        # Discord
        self.client = client

    def run(self) -> None:
        @client.tree.command(name='echo', guild=guild)
        @app_commands.describe(message='Echo back this message')
        async def echo(interaction: discord.Interaction, message: str):
            await interaction.response.defer()
            await asyncio.sleep(3)

            response = await interaction.channel.send(message)
            await asyncio.sleep(1)

            for i in range(10):
                response = await response.edit(content=(response.content + ' ' + message))
                await asyncio.sleep(1)
                
            await interaction.followup.send("**Command:** */echo*")

            # await interaction.response.send_message(message)

        @client.tree.command(name='task', guild=guild)
        @app_commands.describe(message='Write your task here')
        async def chat(interaction: discord.Interaction, message: str):
            await interaction.response.defer()
            response = await api.math_bot_stream(message)

            stream = await interaction.channel.send(content="**Response:**")

            text = ""
            is_new = True
            is_block = False
            for token in response:
                tokens = token.split('\n')

                for t in tokens:
                    if t == '':
                        text = '' if not is_block else text
                        is_new = True

                    elif is_block:
                        is_block = '```' not in t
                        text += ('\n' + t) if is_new else t
                        stream = await stream.edit(content=text)

                    else:
                        is_block = '```' in t

                        if is_new:
                            text = t
                            stream = await interaction.channel.send(content=text)
                        else:
                            text += t
                            is_new = True
                            stream = await stream.edit(content=text)

                if tokens:
                    is_new = not tokens[-1]

            await interaction.followup.send('{user}: "{message}"'.format(
                user=interaction.user.mention,
                message=message
            ))

        self.client.run(DISCORD_TOKEN)
