from queue import Queue
from typing import List, Dict, Union, Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.language_model import BaseLanguageModel

from framework import console
from framework.agent import AgentTool
from framework.research_agent.stream import ResearchAgentStream
from framework.research_agent.keywords import Keyword


step_template = f"""This workflow integrates deterministic algorithms with AI capabilities. 
The purpose of this workflow is to preform a comprehensive research and find insights on a given question.
You have access to the following tools:

{{tool_desc}}

This is the workflow's format:

WORKFLOW:
{Keyword.QUESTION}: the question to research
{Keyword.PROGRESS}: the progress that was in the research made so far 

To use a tool, follow this format:

{Keyword.THOUGHT}: you should always think about what to do
{Keyword.TOOL}: the tool to use, should be one of [{{tool_names}}]
{Keyword.INPUT}: the input for the tool
{Keyword.OBSERVATION}: stop and wait for the output of the tool
...(This {Keyword.THOUGHT}/{Keyword.TOOL}/{Keyword.INPUT}/{Keyword.OBSERVATION} can repeat N times)

If you refer to information from the workflow as part of the input, you must quote it for the tool.
To add an insight to the research, follow this format:

{Keyword.INSIGHT}: the insight to add to the research
{Keyword.REFLECTION}: reflect on the insight you made and give yourself notes 
...(This {Keyword.INSIGHT}/{Keyword.REFLECTION} can repeat N times)

When you are finished with the research, follow this format:

{Keyword.THOUGHT}: I know the final answer
{Keyword.ANSWER}: the final answer with all the insights

Following the format above, complete the workflow below as best as you can.

Begin!

WORKFLOW:
{Keyword.QUESTION}: {{question}}
{Keyword.PROGRESS}: {{progress}}

{{workflow}}"""


class ResearchAgent:
    def __init__(
            self, llm: BaseLanguageModel,
            tools: List[AgentTool],
            max_iterations: int,
            stream: Optional[ResearchAgentStream] = None
    ):

        self.model = llm
        self.tools = tools
        self.max_iter = max_iterations
        self.stream = ResearchAgentStream(queue=Queue(), is_verbose=True) if not stream else stream

        self.question = ''
        self.workflow = ''
        self.progress = ''
        self.answer = ''
        self.tool_names = ','.join([f'{t.name}' for t in self.tools])
        self.tool_desc = '\n'.join([f'{t.name}: {t.description}' for t in self.tools])
        self.tool_map = {t.name: t for t in self.tools}

    def choose_step(self, workflow: str) -> Union[str, None]:
        if f'\n{Keyword.OBSERVATION}:' in workflow and f'\n{Keyword.REFLECTION}:' in workflow:
            observation_index = workflow.find(f'{Keyword.OBSERVATION}:')
            paragraph_index = workflow.find(f'{Keyword.REFLECTION}:')

            if observation_index < paragraph_index:
                return Keyword.OBSERVATION
            else:
                return Keyword.REFLECTION

        elif f'\n{Keyword.OBSERVATION}:' in workflow:
            return Keyword.OBSERVATION

        elif f'\n{Keyword.REFLECTION}:' in workflow:
            return Keyword.REFLECTION

        elif f'\n{Keyword.ANSWER}:' in workflow:
            return Keyword.ANSWER

        return None

    def step(self) -> Union[Dict, None]:
        step_prompt = PromptTemplate.from_template(step_template)
        step_chain = LLMChain(llm=self.model, prompt=step_prompt)
        try:
            output = step_chain(
                inputs={
                    'tool_desc': self.tool_desc,
                    'tool_names': self.tool_names,
                    'question': self.question,
                    'progress': self.progress.strip(),
                    'workflow': self.workflow.strip()
                }
            )
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

            elif decision == Keyword.REFLECTION:
                insight_index = step.find(f'{Keyword.INSIGHT}:')
                reflection_index = step.find(f'{Keyword.REFLECTION}:')

                result[Keyword.INSIGHT] = \
                    step[insight_index + len(f'{Keyword.INSIGHT}:'):reflection_index].strip()
                result[Keyword.REFLECTION] = \
                    step[reflection_index + len(f'{Keyword.REFLECTION}:'):].split('\n')[0].strip()

            elif decision == Keyword.ANSWER:
                answer_index = step.find(f'\n{Keyword.ANSWER}:')
                result[Keyword.ANSWER] = step[answer_index + len(f'\n{Keyword.ANSWER}:'):].strip()

            else:
                return None

            return result

        except Exception as e:
            raise e

    async def process_step(self, data: Dict) -> bool:
        if data is None:
            self.workflow += "\nRemember - Strictly follow the format you were given!\n"
            self.stream.write("\nRemember - Strictly follow the format you were given!\n")

            return False

        elif Keyword.ANSWER in data.keys():
            self.answer = data[Keyword.ANSWER]

            self.workflow += f"\n\n{Keyword.THOUGHT}: I now have enough insights"
            self.stream.write(f"{Keyword.THOUGHT}: I now have enough insights")

            self.workflow += f"\n{Keyword.ANSWER}: {data[Keyword.ANSWER]}"
            self.stream.write(f"{Keyword.ANSWER}: {data[Keyword.ANSWER]}")

            return True

        elif Keyword.REFLECTION in data.keys():
            self.progress += f"\n\n{data[Keyword.INSIGHT]}"
            self.workflow += f"\n{Keyword.INSIGHT}: {data[Keyword.INSIGHT]}"
            self.stream.write(f"{Keyword.INSIGHT}: {data[Keyword.INSIGHT]}")

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

    async def invoke(self, question: str) -> str:
        self.question = question
        self.workflow = ''
        self.progress = ''
        self.answer = ''

        console.bold('\n> ResearchAgent is running\n')
        self.stream.write(f"{Keyword.QUESTION}: {self.question}")

        for i in range(self.max_iter):
            data = self.step()

            try:
                is_done = await self.process_step(data=data)

            except Exception as e:
                self.stream.write(None)
                console.bold('\n> ResearchAgent is exiting due to exception...\n')
                raise e

            if is_done:
                break

            if data is None:
                continue

        self.stream.write(None)
        console.bold('\n> ResearchAgent is finished\n')

        return self.answer
