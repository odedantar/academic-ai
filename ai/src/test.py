import asyncio

from agent_builder import build_agent


async def test():
    agent = build_agent()
    text = """Give me the definition of uniform continuity based on academic books, then write in an organized LaTeX syntax that I can copy and compile my self."""

    return await agent.invoke(text)


if __name__ == '__main__':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test())
