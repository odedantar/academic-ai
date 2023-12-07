import asyncio
import discord
import threading
from queue import Queue
from typing import Awaitable, Callable
from discord import app_commands
from discord.ext import commands

import agents_api as api
from config import DISCORD_TOKEN, GUILD_ID

MESSAGE_MAX_LENGHT = 1900


# Discord API data
intents = discord.Intents.all()
client = commands.Bot(command_prefix='!', intents=intents)
guild = discord.Object(id=GUILD_ID)


# Events
@client.event
async def on_ready():
    await client.tree.sync(guild=guild)
    print(f'{client.user} is now running!')


@client.tree.command(name='echo', guild=guild)
@app_commands.describe(message='Echo back this message')
async def echo(interaction: discord.Interaction, message: str):
    await interaction.response.send_message(message)


@client.tree.command(name='task', guild=guild)
@app_commands.describe(message='Write your task here')
async def chat(interaction: discord.Interaction, message: str):
    async def chat_handler(chat_queue: Queue, interaction: discord.Interaction):
        while True:
            if chat_queue.empty():
                continue

            print("Chat async: Sending message")
            content = chat_queue.get(block=False)

            if not content:
                break

            await interaction.channel.send(content=content)

    chat_queue = Queue()
    stream_queue = Queue()
    event_loop = asyncio.get_event_loop()

    await interaction.response.defer()
    api.math_bot_stream(text=message, stream_queue=stream_queue)

    context = await interaction.channel.send(content="**Response:**")
    print("Discord thread: Starting thread")
    queue_thread = threading.Thread(target=queue_handler, args=[stream_queue, chat_queue])
    queue_thread.start()

    event_loop.create_task(chat_handler(
        chat_queue=chat_queue,
        interaction=interaction
    ))
    await interaction.followup.send('{user}: "{message}"'.format(
        user=interaction.user.mention,
        message=message
    ))


def queue_handler(stream_queue: Queue, chat_queue: Queue):
    text = ''
    leftover = ''
    is_block = False

    while True:
        if stream_queue.empty():
            continue

        print("Discord thread: Reading stream queue...")
        chunk = stream_queue.get(block=False)

        if not chunk:
            break

        chunk = leftover + chunk
        lines = chunk.split('\n')
        leftover = lines.pop()

        print("Discord thread: Entering lines loop...")
        for line in lines:
            if is_block:
                if '```' in line:
                    print("Discord thread: Ending code block")
                    text += '\n' + line
                    is_block = False
                    chat_queue.put(text)
                else:
                    if len(text) >= MESSAGE_MAX_LENGHT:
                        print("Discord thread: Splitting a long code block")
                        chat_queue.put(text + '\n```')
                        text = '```'

                    print("Discord thread: Adding line to code block...")
                    text += '\n' + line

            else:
                text = line
                if '```' in line:
                    print("Discord thread: Starting code block")
                    is_block = True
                elif text != '':
                    chat_queue.put(text)

    if leftover != '':
        print("Discord thread: Putting leftover")
        chat_queue.put(leftover)
    chat_queue.put(None)


def run() -> None:
    client.run(DISCORD_TOKEN)
