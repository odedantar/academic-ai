from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import Tool
from langchain.schema.language_model import BaseLanguageModel

from utilities.tool_wrapper import get_multivariable_chain_tool


solver_template = """As part of your training you've acquired great knowledge, abilities and rigor in a wide 
variety of math fields, except for numerical calculations in which you make constant mistakes and need to use 
a calculator. Use your knowledge and abilities to complete the solution of the math question below:

Question:
Given: {what_is_given}
Task: {math_question}

Remember - Be didactic and rigorous. If you don't have enough knowledge to provide a solution, write "I don't know". 
If you come across a numerical calculation, stop and write: "Calculate: ${{numerical-calculation}}" and give 
instructions to return to you with the results of the calculations to continue the solution. If you've solved the 
question, write at the end of the solution: "Question is solved". Write only the solution and nothing more.

Begin!

Solution:
{partial_solution}"""

variables = {
    "what_is_given": "Given assumptions and known facts which are relevant to the question",
    "math_question": "Clearly stated goal or objective to solve",
    "partial_solution": "(Optional) Part of a solution which is already given and needs completion"
}


def get_question_solver_tool(llm: BaseLanguageModel) -> Tool:
    solver_prompt = PromptTemplate.from_template(solver_template)
    solver_chain = LLMChain(llm=llm, prompt=solver_prompt)

    return get_multivariable_chain_tool(
        llm=llm,
        multivariable_chain=solver_chain,
        variables=variables,
        tool_name="Math question solver",
        tool_description=("Useful for solving one math question at a time. "
                          "PAY ATTENTION - Describe what is given and what is the question with as much "
                          "details as possible")
    )
