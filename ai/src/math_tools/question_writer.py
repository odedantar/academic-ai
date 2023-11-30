from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import Tool
from langchain.schema.language_model import BaseLanguageModel

from utilities.tool_wrapper import get_multivariable_chain_tool


writer_template = """As part of your training you've read vast amounts of advanced math material, and seen 
a wide variety of exercises in different subjects suited for different levels. Use your knowledge and training 
to write a new math exercise relevant to the field of {math_field}, involving {field_subjects}, 
suited for {educational_level} level. Write in the following format:
    
    Question:
    Given: Assumptions and known facts.
    Task: Clearly stated goal or objective.

Here are some examples: 
    
    Question:
    Given: Let A, B, and C be real numbers.
    Task: Prove that If A = B and B = C, then A = C.
    
    Question:
    Given: A rectangular garden needs to be fenced. Two sides are already fenced, and the total length of fencing available is 40 meters.
    Task: Find the dimensions of the garden that maximize its area.
    
    Question:
    Given: Let f(x) = 2x^2 - 3x + 1
    Task: Find the critical points and classify them as maxima, minima, or points of inflection.

Pay attention - Be didactic and rigorous while taking into account the level of the students for whom this exercise is 
meant for. If you don't have enough knowledge to provide a relevant exercise answer with "I don't know". Write 
only the exercise and nothing more. {additional_details}.

Begin!

Question:"""

variables = {
    "math_field": "Field of math of the question",
    "field_subjects": "Specific subjects under the given math_field the question should involve",
    "educational_level": "Level of students for whom the question is meant for",
    "additional_details": "Additional details and instructions for the content of the question"
}


def get_question_writer_tool(llm: BaseLanguageModel) -> Tool:
    writer_prompt = PromptTemplate.from_template(writer_template)
    writer_chain = LLMChain(llm=llm, prompt=writer_prompt)

    return get_multivariable_chain_tool(
        llm=llm,
        multivariable_chain=writer_chain,
        variables=variables,
        tool_name="Math question writer",
        tool_description=("Useful for writing one math questions at a time. "
                          "PAY ATTENTION - Describe what question to write with as much details as possible")
    )
