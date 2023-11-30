import json
from typing import Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import Tool
from langchain.schema.language_model import BaseLanguageModel


wrapper_template = """Here is a documentation of a specific JSON scheme:
JSON: 
{json_scheme}

Fill the fields of this specific JSON scheme according to the following request:
REQUEST: 
{request}

Pay attention - Write only the JSON scheme as you were asked and nothing more.
If a field is optional and you don't have enough information to fill it, you may leave the field empty. 
If a field is non optional and you don't have enough information to fill it, than instead of writing 
the JSON scheme describe what additional information do you need to be able to fill it.

Begin!

JSON:"""


def get_multivariable_chain_tool(
        llm: BaseLanguageModel,
        multivariable_chain: LLMChain,
        variables: dict,
        tool_name: str,
        tool_description: str
) -> Tool:

    def tool_wrapper(request: Optional[str] = None) -> str:
        json_scheme = ',\n\t'.join(['"' + name + '": "' + desc + '"' for name, desc in variables.items()])
        json_scheme = '{\n\t' + json_scheme + '\n}'

        prompt = PromptTemplate.from_template(wrapper_template)
        chain = LLMChain(llm=llm, prompt=prompt)

        # check for the values we have been given
        if not request:
            return """Could not continue with an empty request from the tool"""

        else:
            response = chain(
                inputs={
                    'json_scheme': json_scheme,
                    'request': request
                })
            scheme = response['text']

            try:
                inputs = json.loads(scheme, strict=False)

                response = multivariable_chain(
                    inputs=inputs
                )

                return response['text']

            except ValueError as e:
                return """Failed to parse tool request to tool variables"""

    return Tool(
        name=tool_name,
        func=tool_wrapper,
        description=tool_description
    )
