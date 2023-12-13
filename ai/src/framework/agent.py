import inspect
import asyncio
from queue import Queue
from typing import List, Dict, Union, Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.language_model import BaseLanguageModel

from framework import console
from framework.keywords import Keyword
from framework.agent_tool import AgentTool
from framework.agent_stream import AgentStream


step_template = f"""You are a great decision maker but terrible at anything else.
Answer the following questions as best you can using the following tools:

{{tool_desc}}

Use this format:

{Keyword.QUESTION}: the input question you must answer
{Keyword.THOUGHT}: you should always think about what to do
{Keyword.ACTION}: the action to take, should be one of [{{tool_names}}]
{Keyword.INPUT}: the input to the action
{Keyword.OBSERVATION}: stop and wait for the result of the action
...(This {Keyword.THOUGHT}/{Keyword.ACTION}/{Keyword.INPUT}/{Keyword.OBSERVATION} can repeat N times)

When you have enough information to answer the question, use the following format:

{Keyword.THOUGHT}: I now know the final answer
{Keyword.ANSWER}: the final answer to the original input question

Remember, you're only good at decision making and nothing else. 
Don't attempt to do anything on your own, always use your tools.
When you write the input for the tools, give as much details as are known to you.

Begin!

{Keyword.QUESTION} {{question}}
{{workflow}}
"""


class Agent:
    def __init__(
            self, llm: BaseLanguageModel,
            tools: List[AgentTool],
            max_iterations: int,
            stream: Optional[AgentStream] = None):

        self.model = llm
        self.tools = tools
        self.max_iter = max_iterations
        self.stream = AgentStream(queue=Queue(), is_verbose=True) if not stream else stream

        self.question = ''
        self.workflow = ''
        self.answer = ''
        self.tool_list = [f'{t.name}' for t in self.tools]
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
                    'question': self.question,
                    'workflow': self.workflow
                }
            )
            step = output['text']
        except Exception as e:
            raise e

        result = {}

        try:
            if '\nObservation:' in step:
                thought_index = step.find('Thought:')
                action_index = step.find('Action:')
                action_input_index = step.find('Action Input:')
                observation_index = step.find('Observation:')

                result['thought'] = step[thought_index + len('Thought:'):action_index].strip()
                result['tool'] = step[action_index + len('Action:'):action_input_index].strip()
                result['input'] = step[action_input_index + len('Action Input:'):observation_index].strip()

                if result['tool'] not in self.tool_list:
                    return None

            elif '\nFinal Answer:' in step:
                answer_index = step.find('Final Answer:')
                result['answer'] = step[answer_index + len('Final Answer:'):].strip()

            else:
                return None

            return result

        except Exception as e:
            raise e

    def process_step(self, data: Dict) -> bool:
        if data is None:
            self.workflow += "\n\nCould not parse your thought.\nStick to the format you were given!\n"

            return False

        elif 'answer' in data.keys():
            self.answer = data['answer']

            self.workflow += f"\nThought: I now know the final answer"
            self.stream.write(f"\nThought: I now know the final answer")

            self.workflow += f"\nFinal Answer: {data['answer']}"
            self.stream.write(f"\nFinal Answer: {data['answer']}")

            return True

        else:
            self.workflow += f"\nThought: {data['thought']}"
            self.stream.write(f"\nThought: {data['thought']}")

            self.workflow += f"\nAction: {data['tool']}"
            self.stream.write(f"\nAction: {data['tool']}")

            self.workflow += f"\nAction Input: {data['input']}"
            self.stream.write(f"\nAction Input: {data['input']}")

            return False

    async def invoke(self, question: str) -> str:
        self.question = question
        self.workflow = ''
        self.answer = ''

        console.bold('\n> CustomAgent is running\n')

        self.workflow += f"\nQuestion: {question}"
        self.stream.write(f"\nQuestion: {question}")

        for i in range(self.max_iter):
            data = self.step()
            is_answered = self.process_step(data=data)

            if is_answered:
                break

            try:
                observation = await self.tool_map[data['tool']].invoke(data['input'])
                self.workflow += f"\nObservation: {observation}"
                self.stream.write(f"\nObservation: {observation}")

            except Exception as e:
                self.stream.write(None)
                console.bold('\n> CustomAgent is exiting due to exception...\n')
                raise e

        self.stream.write(None)
        console.bold('\n> CustomAgent is finished\n')

        return self.answer
