from typing import List, Dict, Optional, Union
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import BaseTool, Tool
from langchain.schema.language_model import BaseLanguageModel


step_template = """You are a great planner and decision maker but terrible at anything else.
Answer the following questions as best you can using the following tools:

{tool_list}

Use this format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: stop and wait for the result of the action
...(This Thought/Action/Action Input/Observation can repeat N times)

When you have enough information to answer the question, use the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Remember, you're only good at decision making and nothing else. 
Don't attempt to do anything on your own, always use your tools.
Begin!

Question: {question}
{process}"""


class CustomAgent:
    def __init__(self, llm: BaseLanguageModel, tools: List[BaseTool], max_iterations: int):
        self.model = llm
        self.tools = tools
        self.max_iter = max_iterations

        self.question = ''
        self.process = ''
        self.answer = ''
        self.tool_names = ','.join([f'{t.name}' for t in self.tools])
        self.tool_list = '\n'.join([f'{t.name}: {t.description}' for t in self.tools])
        self.tool_map = {t.name: t for t in self.tools}

    def step(self) -> Union[Dict, None]:
        step_prompt = PromptTemplate.from_template(step_template)
        step_chain = LLMChain(llm=self.model, prompt=step_prompt)

        output = step_chain(
            inputs={
                'tool_list': self.tool_list,
                'tool_names': self.tool_names,
                'question': self.question,
                'process': self.process
            }
        )
        step = output['text']
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

            elif '\nFinal Answer:' in step:
                answer_index = step.find('Final Answer:')
                result['answer'] = step[answer_index + len('Final Answer:'):].strip()

            else:
                result = None

            return result

        except Exception as e:
            raise e

    def run(self, question: str) -> str:
        self.question = question
        self.process = ''
        self.answer = ''

        print(f"\nQuestion: {question}")

        for i in range(self.max_iter):
            data = self.step()

            if data is None:
                self.process = "\nCould not parse the answer.\nStick to the format you were given!"

            elif 'answer' in data.keys():
                self.answer = data['answer']
                self.process += f"\nThought: I now know the final answer"
                self.process += f"\nFinal Answer: {data['answer']}"
                print(f"\nThought: I now know the final answer\nFinal Answer: {data['answer']}")
                break

            try:
                self.process += f"\nThought: {data['thought']}"
                self.process += f"\nAction: {data['tool']}"
                self.process += f"\nAction Input: {data['input']}"
                print(f"\nThought: {data['thought']}\nAction: {data['tool']}\nAction Input: {data['input']}")

                observation = self.tool_map[data['tool']].invoke(data['input'])
                self.process += f"\nObservation: {observation}"
                print(f"\nObservation: {observation}")

            except Exception as e:
                raise e

        return self.answer
