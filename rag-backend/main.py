# from multiprocessing import context
import httpx
from mcp.server.fastmcp import FastMCP
'''
Approach:
    Most bare minimum thing to accomplish before moving forward
    1. store the retrieved huggingface data within chromaDB vector database --> done (query also works)

    2. use the MCP server tweaked approach to reference the chroma db vector database using resources (Suggestion ? Expose the context retrieved via query as a potential tool/resource? not entirely sure as of yet.)

    3. resources should use tools to retrieve the data and send it back to the LLM for context

    4. LLM will then parse the contextual data and formulate an appropriate response to return back to the user on the frontend side.

    5. To build the UI interface, reference streamlit docs to quickly get upto speed with the development.

    For smallest subproblem working model, first get the client and server running.

'''

import chromadb
from datasets import load_dataset
from typing import Any
from pkgutil import get_data

mcp = FastMCP("rag-server")
# TODO : continue here
# sample_dataset_original = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
# chroma_client = chromadb.client()
# collection = chroma_client.create_collection(name="rag_collecton_wiki")
# sample_dataset_alternate = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
# print(sample_dataset_alternate)
#

from chromaDB import ChromaDBVectorDatabase
def get_dataset(dataset_str : str, text_type : str) -> Any:
    return load_dataset(dataset_str, text_type)


# def main():
#     # print("Hello from rag-backend!")
#     # sample_dataset_alternate = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
#     dataset1 = get_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
#     print(dataset1.items())
#     for question in dataset1['test']['question']:
#         print(question)

#     for answers in dataset1['test']['answer']:

# some stuff to play with
# from chromadb import Client

# class VectorDatabase:
#     def __init__(self, collection_name):
#         # Initialize ChromaDB client and create a collection
#         self.client = Client()
#         self.collection = self.client.create_collection(collection_name)

#     def store_data(self, data):
#         """
#         Store a list of dictionaries in the ChromaDB collection.
#         Each dictionary should have 'question', 'answer', and 'id'.
#         """
#         for entry in data:
#             self.collection.add(
#                 documents=[entry["question"], entry["answer"]],
#                 metadatas=[{"id": entry["id"]}],
#                 ids=[str(entry["id"])]  # Ensure IDs are strings
#             )

#     def search(self, query, n_results=5):
#         """
#         Search for the most relevant documents based on the query.
#         Returns the top n_results matching documents.
#         """
#         results = self.collection.query(
#             query=query,
#             n_results=n_results
#         )
#         return results

# Example usage
if __name__ == "__main__":
    # Create an instance of VectorDatabase
    # vector_db = VectorDatabase("my_collection")
    #
    # Data is stored in dictionary format (keys being 'question', 'answer' and 'id')
    data = dict(load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer"))

    # # Store data
    # vector_db.store_data(data)

    # Search for a query
    # query = "What is AI?"
    # results = vector_db.search(query)
    #
    # TODO : fix formatting issues
    #
    for val in list(data['test'])[:3]:
        print(val['question'], val['answer'], val['id'])


    # print(data['test']['question'][:3])
