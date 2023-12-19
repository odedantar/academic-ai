from typing import Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.language_model import BaseLanguageModel

from gpt_utils.models import get_openai_llm


latex_template = """Rewrite the following text in LaTeX:

TEXT: 
{text}

Pay attention - Write only the LaTeX and nothing more.
If there are parts of the text which are properly written in LaTeX, copy them as is.
The result should be a compilable code of a full LaTeX document containing the rewritten text.

Begin!

LATEX:
"""


def get_latex_doc_chain(llm: BaseLanguageModel) -> LLMChain:
    latex_prompt = PromptTemplate.from_template(template=latex_template)
    return LLMChain(llm=llm, prompt=latex_prompt)


def write_latex_document(text: str, llm: Optional[BaseLanguageModel] = None) -> str:
    llm = get_openai_llm(model_name="gpt-4-1106-preview") if llm is None else llm
    latex_chain = get_latex_doc_chain(llm=llm)
    return latex_chain.invoke(input={'text': text})['text']


async def awrite_latex_document(text: str, llm: Optional[BaseLanguageModel] = None) -> str:
    llm = get_openai_llm(model_name="gpt-4-1106-preview") if llm is None else llm
    latex_chain = get_latex_doc_chain(llm=llm)
    return latex_chain.ainvoke(input={'text': text})['text']
