from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.runnable import RunnablePassthrough


proofread_template = """The text below is an attempt at solving a math question. It might be a well written solution, 
but it might as well have some mistakes. Preform a mathematical proofreading to this solution, try to find mathematical 
errors and inconsistencies, fix them if you find any, and write a better solution.
SOLUTION: 
{solution}

Pay attention - Be didactic and rigorous. Write only the solution and nothing more. ALWAYS write in LaTeX code.

Begin!

SOLUTION:
"""


def get_proofread_chain(llm: BaseLanguageModel, input_chain: LLMChain):
    proofread_prompt = PromptTemplate.from_template(template=proofread_template)
    proofread_chain = LLMChain(llm=llm, prompt=proofread_prompt)

    return {'solution': input_chain} | RunnablePassthrough.assign(output=proofread_chain)
