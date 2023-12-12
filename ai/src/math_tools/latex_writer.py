from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import BaseTool
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.runnable import RunnablePassthrough

from utilities.tool_wrapper import code_chain_as_tool


latex_template = """Rewrite the following text in LaTeX syntax.
TEXT: 
{text}

Pay attention - Write only the LaTeX syntax and nothing more.
If there are parts of the text which are properly written in LaTeX, copy them as is.

Begin!

TEXT:
"""

variables = {
    "text": "Text to be typed in LaTeX syntax"
}


def get_latex_chain(llm: BaseLanguageModel):
    latex_prompt = PromptTemplate.from_template(template=latex_template)
    return LLMChain(llm=llm, prompt=latex_prompt)


def get_latex_sequence(llm: BaseLanguageModel, input_chain: LLMChain):
    latex_chain = get_latex_chain(llm=llm)
    return {'text': input_chain} | RunnablePassthrough.assign(output=latex_chain)


def get_latex_tool(
        llm: BaseLanguageModel,
        wrapper_llm: BaseLanguageModel = None
) -> BaseTool:
    latex_chain = get_latex_chain(llm=llm)

    return code_chain_as_tool(
        llm=llm if not wrapper_llm else wrapper_llm,
        chain=latex_chain,
        variables=variables,
        tool_name="LaTeX typer",
        tool_description="Useful for typing a given text in LaTeX syntax",
        language="latex"
    )
