import json
from typing import Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.language_model import BaseLanguageModel

from framework.agent_tool import AgentTool


wrapper_template = """Here is a documentation of a specific JSON scheme:
JSON: 
{json_scheme}

Fill the fields of this specific JSON scheme according to the following request:
REQUEST: 
{request}

Pay attention - Write only the JSON scheme as you were asked and nothing more.
If a field is optional and you don't have enough information to fill it, you may leave it empty. 
If a field is non optional and you don't have enough information to fill it, than instead of writing 
the JSON scheme describe what additional information you need to be able to fill it.

Begin!

JSON:
"""


def chain_as_tool(
        llm: BaseLanguageModel,
        chain: LLMChain,
        variables: dict,
        tool_name: str,
        tool_description: str
) -> AgentTool:

    def wrapper(request: Optional[str] = None) -> str:
        json_scheme = ',\n\t'.join(['"' + name + '": "' + desc + '"' for name, desc in variables.items()])
        json_scheme = '{\n\t' + json_scheme + '\n}'

        wrapper_prompt = PromptTemplate.from_template(wrapper_template)
        wrapper_chain = LLMChain(llm=llm, prompt=wrapper_prompt)

        # check for the values we have been given
        if not request:
            return """Could not continue with an empty request from the tool"""

        response = wrapper_chain(
            inputs={
                'json_scheme': json_scheme,
                'request': request
            })
        scheme = response['text']
        scheme = scheme.replace('```json', '')
        scheme = scheme.replace('```', '')

        try:
            inputs = json.loads(scheme, strict=False)

            output = chain.invoke(inputs)
            return output['text']

        except ValueError as e:
            return """Failed to parse tool request to tool variables. 
            Try editing the input to not include JSON problematic characters like '{' and '}'."""

    return AgentTool(
        function=wrapper,
        name=tool_name,
        description=tool_description
    )


def sequential_chain_as_tool(
        llm: BaseLanguageModel,
        sequential_chain: LLMChain,
        variables: dict,
        tool_name: str,
        tool_description: str
) -> AgentTool:

    def wrapper(request: Optional[str] = None) -> str:
        json_scheme = ',\n\t'.join(['"' + name + '": "' + desc + '"' for name, desc in variables.items()])
        json_scheme = '{\n\t' + json_scheme + '\n}'

        wrapper_prompt = PromptTemplate.from_template(wrapper_template)
        wrapper_chain = LLMChain(llm=llm, prompt=wrapper_prompt)

        # check for the values we have been given
        if not request:
            return """Could not continue with an empty request from the tool"""

        response = wrapper_chain(
            inputs={
                'json_scheme': json_scheme,
                'request': request
            })
        scheme = response['text']
        scheme = scheme.replace('```json', '')
        scheme = scheme.replace('```', '')

        try:
            inputs = json.loads(scheme, strict=False)

            output = sequential_chain.invoke(inputs)
            return output['output']['text']

        except ValueError as e:
            return """Failed to parse tool request to tool variables. 
            Try editing the input to not include JSON problematic characters like '{' and '}'."""

    return AgentTool(
        function=wrapper,
        name=tool_name,
        description=tool_description
    )


def code_chain_as_tool(
        llm: BaseLanguageModel,
        chain: LLMChain,
        variables: dict,
        tool_name: str,
        tool_description: str,
        language: str
) -> AgentTool:

    def wrapper(request: Optional[str] = None) -> str:
        json_scheme = ',\n\t'.join(['"' + name + '": "' + desc + '"' for name, desc in variables.items()])
        json_scheme = '{\n\t' + json_scheme + '\n}'

        wrapper_prompt = PromptTemplate.from_template(wrapper_template)
        wrapper_chain = LLMChain(llm=llm, prompt=wrapper_prompt)

        # check for the values we have been given
        if not request:
            return """Could not continue with an empty request from the tool"""

        response = wrapper_chain(
            inputs={
                'json_scheme': json_scheme,
                'request': request
            })
        scheme = response['text']
        scheme = scheme.replace('```json', '')
        scheme = scheme.replace('```', '')

        try:
            inputs = json.loads(scheme, strict=False)

            output = chain.invoke(inputs)['text']
            output = output.replace(f'```{language}', '')
            output = output.replace('```', '')

            return f"\n```{language}\n{output}\n```\n"

        except ValueError as e:
            return """Failed to parse tool request to tool variables. 
            Try editing the input to not include JSON problematic characters like '{' and '}'."""

    return AgentTool(
        function=wrapper,
        name=tool_name,
        description=tool_description
    )
