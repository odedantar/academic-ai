import os
import copy
import discord

from src.role import Role
from openai import AsyncOpenAI

DISCORD_TOKEN = os.environ['DISCORD_TOKEN']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']


class Chatbot:

    @staticmethod
    async def chat_completion(messages: list, model: str = "gpt-3.5-turbo") -> str:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)

        completion = await client.chat.completions.create(
            model=model,
            messages=messages
        )

        return completion.choices[0].message.content

    def __init__(self):
        self.purpose = ''
        self.conversation = []
        self.is_active = False

    def add_message(self, role: Role, message: str) -> None:
        self.conversation.append({
            "role": role,
            "content": message
        })

    def add_purpose(self, messages: list) -> list:
        purposed = copy.deepcopy(messages)
        purposed.append({
            "role": Role.SYSTEM,
            "content": f'Your purpose: \n {self.purpose}'
        })

        return purposed

    async def get_response(self, message: discord.message) -> str:
        self.add_message(Role.USER, message.content)
        conversation = self.add_purpose(self.conversation)

        for m in conversation:
            print(m)

        response = await self.chat_completion(conversation)
        self.add_message(Role.ASSISTANT, response)

        return response

    def run(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)

        @client.event
        async def on_ready():
            print(f'{client.user} is now running!')

        @client.event
        async def on_message(message):
            username = str(message.author)
            user_message = str(message.content)
            channel = str(message.channel)

            print(f"{username} said: '{user_message}' @ [{channel}]")

            if message.author == client.user:
                return

            if not self.is_active:
                if message.content.startswith(f'!purpose <@{client.user.id}>'):
                    content = str(message.content).replace(f'!purpose <@{client.user.id}>', '')
                    self.purpose = content.strip()
                    self.is_active = True

                    await message.channel.send(f'I am activated, and my purpose is:\n{self.purpose}')

                elif client.user.mentioned_in(message):
                    await message.channel.send('I am inactive, to activate me please state my purpose.'
                                               '\n***Hint:** "!purpose @mention-me write_my_purpose"*')
                return

            else:
                if message.content.startswith(f'!stop <@{client.user.id}>'):
                    self.conversation = []
                    self.is_active = False

                    await message.channel.send('I am deactivated.')

                elif message.content.startswith(f'!repurpose <@{client.user.id}>'):
                    content = str(message.content).replace(f'!repurpose <@{client.user.id}>', '')
                    self.purpose = content.strip()

                    await message.channel.send(f'I am repurposed, my purpose is:\n{self.purpose}')

                else:
                    print("Responding...")
                    response = await self.get_response(message)
                    await message.channel.send(response)

        client.run(DISCORD_TOKEN)
