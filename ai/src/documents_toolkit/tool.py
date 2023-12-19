from gemini_utils.models import get_gemini_llm
from gpt_utils.models import get_openai_llm
from documents_toolkit.structures import STRUCTURES, get_problem_set
from framework.document_agent import DocumentAgent
from search_toolkit.tool import get_search_tool
from documents_toolkit.latex_document import write_latex_document


requirements = """Previous Knowledge: High School Math, Basic Set Theory.
Topics: Sequence Max \ Min and Supremum \ Infimum, The Limit of a Sequence
Notes: Write as rigorous, didactic, and comprehensive as possible.

Document Structure:
{structure}"""

if __name__ == '__main__':
    import asyncio
    loop = asyncio.get_event_loop()

    gemini = get_gemini_llm()
    gpt4 = get_openai_llm(model_name="gpt-4-1106-preview")

    tools = [get_search_tool(llm=gemini)]

    agent = DocumentAgent(
        llm=gpt4,
        tools=tools,
        max_iterations=15
    )

    struct = STRUCTURES[2]
    # struct = get_problem_set(number_of_problems=1)

    requirements = requirements.format(structure=struct.dir())
    print(requirements)
    print()

    draft = loop.run_until_complete(agent.invoke(requirements))
    print(draft)
    print()

    latex = write_latex_document(draft)
    print(latex)

    loop.stop()
