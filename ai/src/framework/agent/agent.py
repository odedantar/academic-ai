from queue import Queue
from typing import List, Dict, Union, Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.language_model import BaseLanguageModel

from framework import console
from framework.agent.tool import AgentTool
from framework.agent.stream import AgentStream
from framework.agent.keywords import Keyword


step_template = f"""This workflow integrates deterministic algorithms with AI capabilities. 
In this workflow you are only responsible for reasoning, decision making, and nothing else.
You have access to the following tools:

{{tool_desc}}

To describe your decision workflow, strictly follow this format:

WORKFLOW:
{Keyword.REQUEST}: the request you must answer
{Keyword.THOUGHT}: you should always think about what to do
{Keyword.TOOL}: the tool to use, should be one of [{{tool_names}}]
{Keyword.INPUT}: the input for the tool
{Keyword.OBSERVATION}: stop and wait for the output of the tool
...(This {Keyword.THOUGHT}/{Keyword.TOOL}/{Keyword.INPUT}/{Keyword.OBSERVATION} can repeat N times)

The tools don't have access to anything from your workflow. 
When you write the tool input you must give as much information as possible.
When you have enough information to answer the request, follow this format:

{Keyword.THOUGHT}: I now know the final answer
{Keyword.ANSWER}: the final answer to the original request

{{clarifications}}
Following the workflow above, answer the request below as best as you can.
Begin!

WORKFLOW:
{Keyword.REQUEST}: {{request}}
{{workflow}}
"""


class Agent:
    def __init__(
            self, llm: BaseLanguageModel,
            tools: List[AgentTool],
            max_iterations: int,
            stream: Optional[AgentStream] = None,
            clarifications: Optional[str] = ""
    ):

        self.model = llm
        self.tools = tools
        self.max_iter = max_iterations
        self.stream = AgentStream(queue=Queue(), is_verbose=True) if not stream else stream
        self.clarifications = clarifications

        self.request = ''
        self.workflow = ''
        self.answer = ''
        self.tool_names = ','.join([f'{t.name}' for t in self.tools])
        self.tool_desc = '\n'.join([f'{t.name}: {t.description}' for t in self.tools])
        self.tool_map = {t.name: t for t in self.tools}

    def step(self) -> Union[Dict, None]:
        step_prompt = PromptTemplate.from_template(step_template)
        step_chain = LLMChain(llm=self.model, prompt=step_prompt)
        try:
            output = step_chain(
                inputs={
                    'tool_desc': self.tool_desc,
                    'tool_names': self.tool_names,
                    'clarifications': self.clarifications,
                    'request': self.request,
                    'workflow': self.workflow
                }
            )
            step = output['text']
        except Exception as e:
            raise e

        result = {}

        try:
            if f'\n{Keyword.OBSERVATION}:' in step:
                thought_index = step.find(f'{Keyword.THOUGHT}:')
                tool_index = step.find(f'{Keyword.TOOL}:')
                input_index = step.find(f'{Keyword.INPUT}:')
                observation_index = step.find(f'{Keyword.OBSERVATION}:')

                result[Keyword.THOUGHT] = step[thought_index + len(f'{Keyword.THOUGHT}:'):tool_index].strip()
                result[Keyword.TOOL] = step[tool_index + len(f'{Keyword.TOOL}:'):input_index].strip()
                result[Keyword.INPUT] = step[input_index + len(f'{Keyword.INPUT}:'):observation_index].strip()

                if result[Keyword.TOOL] not in self.tool_map.keys():
                    return None

            elif f'\n{Keyword.ANSWER}:' in step:
                answer_index = step.find(f'\n{Keyword.ANSWER}:')
                result[Keyword.ANSWER] = step[answer_index + len(f'\n{Keyword.ANSWER}:'):].strip()

            else:
                return None

            return result

        except Exception as e:
            raise e

    def process_step(self, data: Dict) -> bool:
        if data is None:
            self.workflow += "\nThe algorithm could not parse my workflow. " \
                             "I must stick to the format I was given!\n"
            self.stream.write("\nThe algorithm could not parse my workflow. " 
                              "I must stick to the format I was given!\n")

            return False

        elif Keyword.ANSWER in data.keys():
            self.answer = data[Keyword.ANSWER]

            self.workflow += f"\n\n{Keyword.THOUGHT}: I now know the final answer"
            self.stream.write(f"{Keyword.THOUGHT}: I now know the final answer")

            self.workflow += f"\n{Keyword.ANSWER}: {data[Keyword.ANSWER]}"
            self.stream.write(f"{Keyword.ANSWER}: {data[Keyword.ANSWER]}")

            return True

        else:
            self.workflow += f"\n\n{Keyword.THOUGHT}: {data[Keyword.THOUGHT]}"
            self.stream.write(f"{Keyword.THOUGHT}: {data[Keyword.THOUGHT]}")

            self.workflow += f"\n{Keyword.TOOL}: {data[Keyword.TOOL]}"
            self.stream.write(f"{Keyword.TOOL}: {data[Keyword.TOOL]}")

            self.workflow += f"\n{Keyword.INPUT}: {data[Keyword.INPUT]}"
            self.stream.write(f"{Keyword.INPUT}: {data[Keyword.INPUT]}")

            return False

    async def invoke(self, request: str) -> str:
        self.request = request
        self.workflow = ''
        self.answer = ''

        console.bold('\n> CustomAgent is running\n')
        self.stream.write(f"{Keyword.REQUEST}: {self.request}")

        for i in range(self.max_iter):
            data = self.step()
            is_answered = self.process_step(data=data)

            if is_answered:
                break

            if data is None:
                continue

            try:
                observation = await self.tool_map[data[Keyword.TOOL]].invoke(data[Keyword.INPUT])
                self.workflow += f"\n{Keyword.OBSERVATION}: {observation}"
                self.stream.write(f"{Keyword.OBSERVATION}: {observation}")

            except Exception as e:
                self.stream.write(None)
                console.bold('\n> CustomAgent is exiting due to exception...\n')
                raise e

        self.stream.write(None)
        console.bold('\n> CustomAgent is finished\n')

        return self.answer
