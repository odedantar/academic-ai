from __future__ import annotations

import asyncio
from typing import List, Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.language_model import BaseLanguageModel

from documents_toolkit.writing_utils import get_writer_agent


section_request = """Based on the following abstract and partial content of a document: 

ABSTRACT:
{abstract}

PARTIAL DOCUMENT: 
{partial_document}

Attempt to recreate the following section of the document as accurately and comprehensively as possible.
All that is known about the section is what was its name and description:

NAME: 
{name}

DESCRIPTION: 
{description}

Use your retrieval tool to fill in the gaps about anything you need additional information about.
Write only the section and nothing else.
"""

draft_template = """Based on the following abstract and structure, 
attempt to recreate the full document as accurately and comprehensively as possible:

ABSTRACT:
{abstract}

STRUCTURE: 
{structure}

Pay attention - Strictly follow the given document structure. 
Within the given structure you have creative freedom as long as you follow the guidelines of the given abstract. 
Begin!

DOCUMENT:
"""


class DocumentStructure:

    def __init__(self, name: str, description: str, structure: Optional[List[DocumentStructure]] = None):

        self.name = name
        self.description = description
        self.structure = structure

    def indented_list(self) -> List[str]:

        structure_list = [f"* {self.name}\n> {self.description}"]

        if self.structure is None:
            return structure_list

        for doc in self.structure:
            indented = []
            for section in doc.indented_list():
                indented.append('\n'.join([f"\t{line}" for line in section.split('\n')]))

            structure_list += indented

        return structure_list

    def dir(self):

        return '\n\n'.join(self.indented_list())

    async def draft_list(self, abstract: str, partial_list: Optional[List[str]] = None) -> List[str]:

        if self.structure is None:
            writer = get_writer_agent()
            request = section_request.format(
                abstract=abstract,
                partial_document="\n".join(partial_list) if partial_list is not None else "empty",
                name=self.name,
                description=self.description
            )

            return [await writer.invoke(request)]

        section_list = []
        for struct in self.structure:
            section_list += await struct.draft_list(abstract, section_list)

        return section_list

    async def adraft(self, abstract: str) -> str:

        section_list = await self.draft_list(abstract)
        return "\n".join(section_list)

    def write_draft(self, llm: BaseLanguageModel, abstract: str) -> str:

        draft_prompt = PromptTemplate.from_template(draft_template)
        draft_chain = LLMChain(llm=llm, prompt=draft_prompt)

        structure = self.dir()

        draft = draft_chain(
            inputs={
                'abstract': abstract,
                'structure': structure
            }
        )

        return draft['text']
