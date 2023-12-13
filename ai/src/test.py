import asyncio

from agent_builder import build_agent


async def test():
    agent = build_agent()
    text = """Give me the definition of uniform continuity based on academic books and then type that definition in LaTeX syntax."""

    return await agent.invoke(text)


if __name__ == '__main__':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test())
