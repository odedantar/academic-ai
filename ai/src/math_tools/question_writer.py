from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import Tool
from langchain.schema.language_model import BaseLanguageModel

from math_tools.latex_transformer import get_latex_chain
from utilities.tool_wrapper import get_multivariable_chain_tool


writer_template = """Write a new math question relevant to the field of {math_field}, involving {field_subjects}, 
suited for {educational_level} level. Write in the following format:
    
Question:
Given: Assumptions and known facts.
Task: Clearly stated goal or objective.

For example: 
    
Question:
Given: Let A, B, and C be real numbers.
Task: Prove that If A = B and B = C, then A = C.

Remember - Be didactic and rigorous while considering the level of the students for whom this question is 
intended. If you don't have enough knowledge to provide a question, answer "I don't know". Write only 
the exercise and nothing more. Write in LaTeX code. {additional_details}.

Begin!

Question:
"""

variables = {
    "math_field": "Field of math of the question",
    "field_subjects": "Specific subjects under the given math_field the question should involve",
    "educational_level": "Level of students for whom the question is meant for",
    "additional_details": "Additional details and instructions for the writing of the question"
}


def get_question_writer_tool(
        llm: BaseLanguageModel,
        latex_llm: BaseLanguageModel = None,
        wrapper_llm: BaseLanguageModel = None
) -> Tool:
    writer_prompt = PromptTemplate.from_template(template=writer_template)
    writer_chain = get_latex_chain(
        llm=llm if not latex_llm else latex_llm,
        input_chain=LLMChain(llm=llm, prompt=writer_prompt)
    )

    return get_multivariable_chain_tool(
        llm=llm if not wrapper_llm else wrapper_llm,
        multivariable_chain=writer_chain,
        variables=variables,
        tool_name="Math question writer",
        tool_description=("Useful for WRITING math questions, one at a time. "
                          "PAY ATTENTION - Describe what question to write with as much details as possible")
    )
