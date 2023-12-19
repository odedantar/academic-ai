from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.language_model import BaseLanguageModel

from framework.agent import AgentTool
from math_toolkit.latex_writer import get_latex_sequence
from framework.chain_wrappers import sequential_chain_as_tool


solver_template = """As part of your training you've acquired great knowledge, abilities and rigor in a wide 
variety of math fields, except for numerical calculations in which you make constant mistakes and need to use 
a calculator. Use your abilities to solve the math question below:

Question:
Given: {what_is_given}
Task: {math_question}

Remember - Be didactic and rigorous. If you don't have enough info to solve the question, write "I don't know". 
If you come across a numerical calculation, stop and write: "Calculate: ${{numerical-calculation}}" and give 
instructions on how to continue the solution. Write only the solution and nothing more. If you've solved the 
question, write at the end of the solution: "Question is solved". Write in LaTeX code.

Begin!

Solution:
"""

variables = {
    "what_is_given": "Given assumptions and known facts which are relevant to the question",
    "math_question": "Clearly stated goal or objective to solve"
}


def get_question_solver_tool(
        llm: BaseLanguageModel,
        latex_llm: BaseLanguageModel = None,
        wrapper_llm: BaseLanguageModel = None
) -> AgentTool:

    solver_prompt = PromptTemplate.from_template(solver_template)
    solver_chain = get_latex_sequence(
        llm=llm if not latex_llm else latex_llm,
        input_chain=LLMChain(llm=llm, prompt=solver_prompt)
    )

    return sequential_chain_as_tool(
        llm=llm if not wrapper_llm else wrapper_llm,
        sequential_chain=solver_chain,
        variables=variables,
        tool_name="Math question solver",
        tool_description=("Useful for SOLVING math questions, one at a time. "
                          "PAY ATTENTION - Describe what is given and what is the question with as much "
                          "details as possible")
    )
