from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import Tool
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.runnable import RunnablePassthrough

from math_tools.latex_transformer import get_latex_chain
from utilities.tool_wrapper import get_multivariable_chain_tool


proofreader_template = """The text below is an attempt at mathematical writing. It might be well written, 
but it might as well have some mistakes. Preform a mathematical proofreading to this text, focus on 
mathematical errors and inconsistencies. If the given text is well written without mistakes, copy it as is. 
If you find mistakes, write a better draft with the the necessary corrections applied.

TEXT: 
{math_text}

Remember - Be didactic and rigorous. Write only the final solution and nothing more. Write in LaTeX code.
{additional_details}.

Begin!

TEXT:
"""

variables = {
    "math_text": "Mathematical solution to proofread",
    "additional_details": "Additional details and instructions for the proofreading"
}


def get_proofreader_tool(
        llm: BaseLanguageModel,
        latex_llm: BaseLanguageModel = None,
        wrapper_llm: BaseLanguageModel = None
) -> Tool:
    proofreader_prompt = PromptTemplate.from_template(template=proofreader_template)
    proofreader_chain = get_latex_chain(
        llm=llm if not latex_llm else latex_llm,
        input_chain=LLMChain(llm=llm, prompt=proofreader_prompt)
    )

    return get_multivariable_chain_tool(
        llm=llm if not wrapper_llm else wrapper_llm,
        multivariable_chain=proofreader_chain,
        variables=variables,
        tool_name="Math proofreader",
        tool_description=("Useful for PROOFREADING mathematical texts, one at a time. "
                          "PAY ATTENTION - Give as much details as possible about the text.")
    )