"""
Useful documentation:
https://python.langchain.com/docs/integrations/vectorstores/faiss
https://medium.com/@ahmed.mohiuddin.architecture/using-ai-to-chat-with-your-documents-leveraging-langchain-faiss-and-openai-3281acfcc4e9
https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed
"""

import asyncio
import threading
from queue import Queue
from typing import List, Union, Optional
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from config import VECTOR_STORE_PATH


class CustomVectorStore:
    def __init__(self, path_to_index: Optional[str] = None, is_initialized: bool = True):
        self.is_init = is_initialized
        self.path = VECTOR_STORE_PATH if not path_to_index else path_to_index
        self.embeddings = OpenAIEmbeddings()

        if self.is_init:
            self.store = FAISS.load_local(self.path, self.embeddings)
        else:
            self.store = None

    def save(self):
        self.store.save_local(self.path)

    def insert_pdf(self, pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n")
        docs = text_splitter.split_documents(documents=documents)
        pdf_store = FAISS.from_documents(docs, self.embeddings)

        if not self.is_init:
            self.store = pdf_store
            self.is_init = True
        else:
            self.store.merge_from(pdf_store)

        self.save()

    async def ainsert_pdf(self, pdf_path: str):
        raise NotImplementedError("Async insert_pdf is not yet implemented")

    def search(self, query: str, k: Optional[int] = 3) -> List[Document]:
        return self.store.similarity_search(query=query, k=k)

    async def asearch(self, query: str, k: Optional[int] = 3) -> List[Document]:

        def preform_query(query: str, k: int, document_queue: Queue[Union[Document, None]]):
            documents = self.store.similarity_search(query=query, k=k)
            for d in documents:
                document_queue.put(d)

            document_queue.put(None)

        document_queue: Queue[Union[Document, None]] = Queue()
        worker = threading.Thread(target=preform_query, args=[query, k, document_queue])
        worker.start()

        document = {}
        documents = []
        while document is not None:
            if document_queue.empty():
                await asyncio.sleep(1)
                continue

            document = document_queue.get(block=False)
            documents.append(document)

        return documents


def insert_books(paths: List, is_initialized: bool):
    store = CustomVectorStore(is_initialized=is_initialized)
    for path in paths:
        print(f"Inserting: {path}")
        store.insert_pdf(pdf_path=path)


if __name__ == '__main__':
    insert_books([
        # Mathematics:

        # "books/Linear Algebra Done Right - Sheldon Axler - 2015.pdf",
        # "books/Understanding Analysis - Stephen Abbott - 2001.pdf",
        # "books/Discrete Mathematics - B. S. Vatsa - 2009.pdf",
        # "books/Combinatorics 1 - Peter Cameron - Lecture Notes - 2014.pdf",
        # "books/Complex Function Theory - Donald Sarason - 2007.pdf",
        # "books/Topology - James R. Munkres - 2014.pdf",

        # Physics:

        # "books/Introduction to Mechanics - Kleppner D. Kolenkow R.J. - 2014.pdf",
        # "books/Electricity and Magnetism - Purcell E.M., Morin D.J. - 2013.pdf",
        # "books/Thermodynamics - Enrico Fermi - 2012.pdf",
        # "books/The Feynman Lectures on Physics - Volume 1.pdf",
        # "books/The Feynman Lectures on Physics - Volume 2.pdf",
        # "books/The Feynman Lectures on Physics - Volume 3.pdf"
    ], is_initialized=True)
