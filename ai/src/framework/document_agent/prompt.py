SCHEME = """{{
    "{requirements}": "the requirements for the writing of the document",
    "{progress}": "the progress of the writing that was made so far",
    "workflow": [
        # To use a tool, follow this format:
        {{
            "{tool}": "the tool to use, should be one of [{tool_names}]",
            "{input}": "the input for the tool",
            "{observation}": "the output of the tool"
        }},
        # ...(This scheme can repeat N times)
        # If you refer to information from the workflow, you must quote it for the tool.
        
        # To add a paragraph to the document, follow this format:
        {{
            "{paragraph}": "the paragraph to add to the document"
        }},
        # ...(This scheme can repeat N times)

        # When you are finished writing the document, follow this format:
        {{
            "{document}": "the final document that matches the requirements"
        }}
    ]
}}"""

TEMPLATE = """This workflow integrates deterministic algorithms with AI capabilities. 
The purpose of this workflow is to write a document based on pre-given requirements.
You have access to the following tools:

{tool_desc}

This is the JSON scheme of the workflow:

WORKFLOW:
{{
    "{progress}": "the progress of the writing that was made so far",
    "workflow": [
        # To use a tool, follow this format:
        {{
            "{tool}": "the tool to use, should be one of [{tool_names}]",
            "{input}": "the input for the tool",
            "{observation}": "the output of the tool"
        }},
        # ...(This scheme can repeat N times)
        # If you refer to information from the workflow, you must quote it for the tool.
        
        # To add a paragraph to the document, follow this format:
        {{
            "{paragraph}": "the paragraph to add to the document"
        }},
        # ...(This scheme can repeat N times)

        # When you are finished writing the document, follow this format:
        {{
            "{document}": "the final document that matches the requirements"
        }}
    ]
}}

Following the JSON scheme above, complete the scheme below to match the requirements as best as you can.
Write only the completion of the scheme and nothing else.

Begin!

REQUIREMENTS:
{requirements_input}

WORKFLOW:
{{
    "{progress}": {progress_input},
    "workflow": [
        {workflow}
        # Remember - Strictly follow the JSON scheme above!
"""
