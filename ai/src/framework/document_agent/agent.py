import json
from queue import Queue
from typing import List, Dict, Union, Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.language_model import BaseLanguageModel

from framework import console
from framework.agent import AgentTool
from framework.document_agent.prompt import SCHEME, TEMPLATE
from framework.document_agent.stream import DocumentAgentStream
from framework.document_agent.keywords import Keyword


step_template = f"""This workflow integrates deterministic algorithms with AI capabilities. 
The purpose of this workflow is to write a document based on pre-given requirements.
You have access to the following tools:

{{tool_desc}}

This is the workflow's format:

WORKFLOW:
{Keyword.REQUIREMENTS}: the requirements for the writing of the document
{Keyword.PROGRESS}: the progress of the writing that was made so far

To use a tool, follow this format:

{Keyword.THOUGHT}: you should always think about what to do
{Keyword.TOOL}: the tool to use, should be one of [{{tool_names}}]
{Keyword.INPUT}: the input for the tool
{Keyword.OBSERVATION}: stop and wait for the output of the tool
...(This {Keyword.THOUGHT}/{Keyword.TOOL}/{Keyword.INPUT}/{Keyword.OBSERVATION} can repeat N times)

If you refer to information from the workflow, you must quote it for the tool.
To add a paragraph to the document, follow this format:

{Keyword.THOUGHT}: you should always think about what to write
{Keyword.PARAGRAPH}: the paragraph to add to the document
{Keyword.REFLECTION}: reflect on what you wrote and give yourself notes 
...(This {Keyword.THOUGHT}/{Keyword.PARAGRAPH}/{Keyword.REFLECTION} can repeat N times)

When you are finished writing the document, follow this format:

{Keyword.THOUGHT}: I now have the final document
{Keyword.DOCUMENT}: the final document that matches the requirements

Following the format above, complete the workflow below as best as you can.

Begin!

WORKFLOW:
{Keyword.REQUIREMENTS}: {{requirements}}
{Keyword.PROGRESS}: {{progress}}

{{workflow}}

Remember - Always try to match the requirements as best as you can.
"""


class DocumentAgent:
    def __init__(
            self, llm: BaseLanguageModel,
            tools: List[AgentTool],
            max_iterations: int,
            stream: Optional[DocumentAgentStream] = None
    ):

        self.model = llm
        self.tools = tools
        self.max_iter = max_iterations
        self.stream = DocumentAgentStream(queue=Queue(), is_verbose=True) if not stream else stream

        self.requirements = ''
        self.workflow = ''
        self.progress = ''
        # self.progress = []
        self.document = ''
        self.tool_names = ','.join([f'{t.name}' for t in self.tools])
        self.tool_desc = '\n'.join([f'{t.name}: {t.description}' for t in self.tools])
        self.tool_map = {t.name: t for t in self.tools}

    def choose_step(self, workflow: str) -> Union[str, None]:
        if f'\n{Keyword.OBSERVATION}:' in workflow and f'\n{Keyword.PARAGRAPH}:' in workflow:
            observation_index = workflow.find(f'{Keyword.OBSERVATION}:')
            paragraph_index = workflow.find(f'{Keyword.PARAGRAPH}:')

            if observation_index < paragraph_index:
                return Keyword.OBSERVATION
            else:
                return Keyword.PARAGRAPH

        elif f'\n{Keyword.OBSERVATION}:' in workflow:
            return Keyword.OBSERVATION

        elif f'\n{Keyword.PARAGRAPH}:' in workflow:
            return Keyword.PARAGRAPH

        elif f'\n{Keyword.DOCUMENT}:' in workflow:
            return Keyword.DOCUMENT

        return None

    def step(self) -> Union[Dict, None]:
        step_prompt = PromptTemplate.from_template(step_template)
        # step_prompt = PromptTemplate.from_template(TEMPLATE)
        step_chain = LLMChain(llm=self.model, prompt=step_prompt, verbose=True)
        try:
            output = step_chain(
                inputs={
                    'tool_desc': self.tool_desc,
                    'tool_names': self.tool_names,
                    'requirements': self.requirements,
                    'progress': self.progress.strip(),
                    'workflow': self.workflow.strip()
                }
            )

            # output = step_chain(
            #     inputs={
            #         'tool_desc': self.tool_desc,
            #         'workflow_scheme': SCHEME,
            #         'requirements': Keyword.REQUIREMENTS,
            #         'progress': Keyword.PROGRESS,
            #         'thought': Keyword.THOUGHT,
            #         'tool': Keyword.TOOL,
            #         'input': Keyword.INPUT,
            #         'observation': Keyword.OBSERVATION,
            #         'paragraph': Keyword.PARAGRAPH,
            #         'reflection': Keyword.REFLECTION,
            #         'document': Keyword.DOCUMENT,
            #         'tool_names': self.tool_names,
            #         'requirements_input': self.requirements,
            #         # 'progress_input': self.progress.strip(),
            #         'progress_input': '[{progress}]'.format(progress=',\n'.join(self.progress).strip()),
            #         'workflow': self.workflow.strip()
            #     }
            # )
            step = output['text']
        except Exception as e:
            raise e

        result = {}
        decision = self.choose_step(step)

        try:
            if decision == Keyword.OBSERVATION:
                thought_index = step.find(f'{Keyword.THOUGHT}:')
                tool_index = step.find(f'{Keyword.TOOL}:')
                input_index = step.find(f'{Keyword.INPUT}:')
                observation_index = step.find(f'{Keyword.OBSERVATION}:')

                result[Keyword.THOUGHT] = step[thought_index + len(f'{Keyword.THOUGHT}:'):tool_index].strip()
                result[Keyword.TOOL] = step[tool_index + len(f'{Keyword.TOOL}:'):input_index].strip()
                result[Keyword.INPUT] = step[input_index + len(f'{Keyword.INPUT}:'):observation_index].strip()

                if result[Keyword.TOOL] not in self.tool_map.keys():
                    return None

            elif decision == Keyword.PARAGRAPH:
                thought_index = step.find(f'{Keyword.THOUGHT}:')
                paragraph_index = step.find(f'{Keyword.PARAGRAPH}:')
                reflection_index = step.find(f'{Keyword.REFLECTION}:')

                result[Keyword.THOUGHT] = \
                    step[thought_index + len(f'{Keyword.THOUGHT}:'):paragraph_index].strip()
                result[Keyword.PARAGRAPH] = \
                    step[paragraph_index + len(f'{Keyword.PARAGRAPH}:'):reflection_index].strip()
                result[Keyword.REFLECTION] = \
                    step[reflection_index + len(f'{Keyword.REFLECTION}:'):].split('\n')[0].strip()

            elif decision == Keyword.DOCUMENT:
                document_index = step.find(f'\n{Keyword.DOCUMENT}:')
                result[Keyword.DOCUMENT] = step[document_index + len(f'\n{Keyword.DOCUMENT}:'):].strip()

            else:
                return None

            return result

        except Exception as e:
            raise e

    async def process_step(self, data: Dict) -> bool:
        if data is None:
            self.workflow += "\n# Remember - Strictly follow the format above!\n"
            self.stream.write("\n# Remember - Strictly follow the format above!\n")

            return False

        elif Keyword.DOCUMENT in data.keys():
            self.document = data[Keyword.DOCUMENT]

            self.workflow += f"\n\n{Keyword.THOUGHT}: I now have the final document"
            self.stream.write(f"{Keyword.THOUGHT}: I now have the final document")

            self.workflow += f"\n{Keyword.DOCUMENT}: {data[Keyword.DOCUMENT]}"
            self.stream.write(f"{Keyword.DOCUMENT}: {data[Keyword.DOCUMENT]}")

            return True

        elif Keyword.PARAGRAPH in data.keys():
            self.workflow += f"\n\n{Keyword.THOUGHT}: {data[Keyword.THOUGHT]}"
            self.stream.write(f"{Keyword.THOUGHT}: {data[Keyword.THOUGHT]}")

            self.progress += f"\n\n{data[Keyword.PARAGRAPH]}"
            self.workflow += f"\n{Keyword.PARAGRAPH}: {data[Keyword.PARAGRAPH]}"
            self.stream.write(f"{Keyword.PARAGRAPH}: {data[Keyword.PARAGRAPH]}")

            self.workflow += f"\n{Keyword.REFLECTION}: {data[Keyword.REFLECTION]}"
            self.stream.write(f"{Keyword.REFLECTION}: {data[Keyword.REFLECTION]}")

            return False

        else:
            self.workflow += f"\n\n{Keyword.THOUGHT}: {data[Keyword.THOUGHT]}"
            self.stream.write(f"{Keyword.THOUGHT}: {data[Keyword.THOUGHT]}")

            self.workflow += f"\n{Keyword.TOOL}: {data[Keyword.TOOL]}"
            self.stream.write(f"{Keyword.TOOL}: {data[Keyword.TOOL]}")

            self.workflow += f"\n{Keyword.INPUT}: {data[Keyword.INPUT]}"
            self.stream.write(f"{Keyword.INPUT}: {data[Keyword.INPUT]}")

            try:
                observation = await self.tool_map[data[Keyword.TOOL]].invoke(data[Keyword.INPUT])
            except Exception as e:
                raise e

            self.workflow += f"\n{Keyword.OBSERVATION}: {observation}"
            self.stream.write(f"{Keyword.OBSERVATION}: {observation}")

            return False

    async def invoke(self, requirements: str) -> str:
        self.requirements = requirements
        self.workflow = ''
        self.progress = ''
        self.document = ''

        console.bold('\n> DocumentAgent is running\n')
        self.stream.write(f"{Keyword.REQUIREMENTS}:\n{self.requirements}")

        for i in range(self.max_iter):
            data = self.step()

            try:
                is_done = await self.process_step(data=data)

            except Exception as e:
                self.stream.write(None)
                console.bold('\n> DocumentAgent is exiting due to exception...\n')
                raise e

            if is_done:
                break

            if data is None:
                continue

        self.stream.write(None)
        console.bold('\n> DocumentAgent is finished\n')

        return self.document
