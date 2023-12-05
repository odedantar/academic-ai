from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.runnable import RunnablePassthrough


latex_template = """The following text is an attempt at mathematical writing. Rewrite it in LaTeX code.
TEXT: 
{math_text}

Pay attention - Write only the LaTeX code and nothing more.
If there are parts of the text which are properly written in LaTeX, copy them as is.

Begin!

TEXT:
"""


def get_latex_chain(llm: BaseLanguageModel, input_chain: LLMChain):
    latex_prompt = PromptTemplate.from_template(template=latex_template)
    latex_chain = LLMChain(llm=llm, prompt=latex_prompt)

    return {'math_text': input_chain} | RunnablePassthrough.assign(output=latex_chain)
