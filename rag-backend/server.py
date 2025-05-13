# type:ignore

# TODO : write and connect streamlit UI to the FastAPI server that is hooked to MCP client which is hooked to MCP server.
# TODO : look into proper saving logic for context

# TODO : rename after all relevant testing has been done (this is the code for primary server)
# echo.py
# to simplify, we connect the server to a client manually based on the path
#
# IMPORTANT : https://www.youtube.com/watch?v=3K39NJbp2IA (reference this video to understand)
# additional reference : https://github.com/NarimanN2/openai-playground/tree/main/mcp-agent
# https://www.youtube.com/watch?v=Ln-Tgz8Pmek --> this might help too? (mainly to understand the connection process of streamlit host with mcp server)
#
# Lastly reference the youtube link below (with some minor modification to the server portion of the code (i.e. the tools that I am working with, it should all work intended with FastAPI endpoint )) --> https://www.youtube.com/watch?v=mhdGVbJBswA --> found this to be most helpful
'''

https://block.github.io/goose/docs/tutorials/custom-extensions --> you can even try this for bare minimum working version.

Roadmap : write up the server that will reference the vector database to retrieve relevant contextual information

setup the regular mcp server and client interaction that is going on under the hood and then setup an additional endpoint like /userQuery which will bridge the connection to the stdio server client connection

relevant github file for reference : https://github.com/alejandro-ao/mcp-client-python/blob/master/api/main.py

'''

# IMPORTANT NOTE : There's no "tag" keyword

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base
from fastmcp import Client
from chromaDB import client, ChromaDBVectorDatabase, get_huggingface_data
from datetime import datetime
from typing import Any, List, Dict, Union
import anthropic
import chromadb
import asyncio

# helper functions
def check_collection_data_count(collection_name : str) -> dict[str, Any]:
    '''
    if it's a newly created collection, the count will be zero
    '''
    chroma_collection = client.get_or_create_collection(name=collection_name,
        metadata={
            "description" : "chroma db vector collection",
            "created" : str(datetime.now())
        }
    )

    # NOTE : chroma_collection isn't really needed in this scenario
    return {
        "collection_count" : chroma_collection.count(),
        "collection" : chroma_collection
    }

mcp = FastMCP(
    name="Rag-Chatbot-Server",
    port=8081,
    host="127.0.0.1",   # set default SSE host
    log_level="DEBUG",
    on_duplicate_tools="warn"   # Warn if tools with the same name are registered (options: 'error', 'warn', 'ignore')
)

# TODO : remove after
# TODO : some of the tools are referencing client but the logic is also built in within the ChromaDBVectorDatabase class, so during refactoring process, rewrite it in terms of the class rather than the global client for cleaner code
#
# define list of relevant tools
@mcp.tool(description="A simple echo tool")
def echo(message: str) -> str:
    return f"Echo: {message}"

@mcp.tool(
    name="context_retriever",
    description="seaches chroma DB to retrieve relevant context and allows control over number of relevant context user wants to retrieve (default : 3) of a particular collection. If the collection doesn't exist, new data will be created and inserted before search query is performed."
)
def retrieve_relevant_context(user_query : str = "", number_of_relevant_context : int = 3, name_of_collection : str = "complete_collection"):
    collection_instance = ChromaDBVectorDatabase(name_of_collection, client)
    if client.get_collection(name=name_of_collection).count() == 0:
        enter_data_to_new_collection(name_of_collection)
    try:
        query_results = collection_instance.search(user_query, number_of_relevant_context)
        return f"Query results are : \n {query_results}"
        # else:
        #     return "Collection does not exist"
    except Exception as e:
        print(f"error occured : {e}")
        return f"error message : {e}"

@mcp.tool(
    name="peek_at_database",
    description="allows for users to retrieve the topmost levels of data. (Default : 3) from the collection you want to retrieve from (default collection name : complete_collection)."
)
def get_topmost_data(number_of_rows : int
    = 3, name_of_collection : str = "complete_collection"):
    try:
        collection_instance = client.get_collection(name=name_of_collection)
        return collection_instance.peek(limit=number_of_rows)
    except Exception as e:
        return f"Failed to retrieve topmost data due to : {e}"

@mcp.tool(
    name="modify_collection_name",
    description="allows user to modify the name of an existing collection"
)
def modify_existing_collection(original_collection : str, new_collection_name : str):
    try:
        current_collection = client.get_collection(original_collection)
        current_collection.modify(name=new_collection_name)
        return f"successfully changed {original_collection} to {new_collection_name}"
    except Exception as e:
        return f"Failed to change collection name due to {e}"

@mcp.tool(
    name="get_list_of_collections",
    description="allows for retrieval of list of availble collections"
)
def get_collection_list() -> Union[str, Any]:
    try:
        return client.list_collections()
    except Exception as e:
        return f"Failed to retrieve list of collections due to {e}"

@mcp.tool(
    name="delete_collection_by_name",
    description="delete a particular collection based on the provided name"
)
def delete_collection_by_name(collection_name : str):
    client.delete_collection(name=collection_name)

@mcp.tool(
    name="enter_data",
    description="This tool will re-enter fresh batch of data on a newly created collection."
)
def enter_data_to_new_collection(collection_name : str) -> str:
    collection_info = check_collection_data_count(collection_name)
    load_data = get_huggingface_data()

    try:

        if collection_info["collection_count"] == 0 and load_data["status_code"] == 200:
            chroma_instance = ChromaDBVectorDatabase(collection_name, client)
            chroma_instance.store_data(load_data["data"])
            return f"Successfully loaded data into collection {collection_name}"

        elif collection_info["collection_count"] > 0:
            return f"{collection_name} already contains data of size {collection_info["collection_count"]}."

        elif load_data["status_code"] == 503:
            return f"Failed to load huggingface data due to {load_data["message"]}."
        else:
            return "Tool failed to execute due to some unknown error. Please try again."
    except Exception as e:
        return f"Error occured due to {e}"

@mcp.tool(
    name="get_collection_data_count",
    description="returns the number of data contained within a particular collection"
)
def get_collection_data_count(name_of_collection : str) -> int:
    return client.get_collection(name=name_of_collection.strip().replace(" ", "")).count()

# TODO : look into ways to reduce the size of the description
@mcp.tool(
    name="get_user_query_history",
    description="""the user queries alongside llm response for the current session is stored within the chroma db collection 'contextual_data'. Can be used to search and retrieve the relevant data stored here for follow-up queries from the user for query history. If your unsure of the user query, use this tool to retrieve previous query related contextual information before attempting to answer. Keep responses brief and utilize previous conversation history stored within the 'contextual_data' to formulate your responses.
    """
)
def retrieve_user_query_history(user_query:str, collection_name : str="contextual_data", n_results:int=5):
    return client.get_collection(collection_name).query(
        query_texts=[user_query],
        n_results=n_results
    )[:100]


# define list of relevant prompts
@mcp.prompt(
    name="convert",
    description="Given the vector data which represents relevant contextual information, use it to formulate an appropriate response that will be sent back to the user, use appropriate easy to read markdown format."
)
def convert_to_NLP(vector_data:str="placeholder data") -> str | Any:
    return str([
        {
            "role" : "system",
            "content" : "You are a helpful assistant skilled at using vector data to formulate human readable responses to user queries. Also make sure to keep track of previous conversation history for follow up responses."
        },
        {
            "role" : "user",
            "content" : f"use the following data for additional context:\n {vector_data}, discard unneccesary meatadata."
        }
    ])

@mcp.prompt(
    name="track_context_history",
    description="fetch previous query related history from the collection 'contextual_data'."
)
def fetch_conversation_history(query : str) -> list[base.Message]:
    return [
        base.UserMessage("Within the collection 'contextual_data', refer to the role of 'user' for previous user queries, and 'assistant' for previous LLM responses."),
        base.UserMessage(query),
        base.AssistantMessage("I will use the current and previous conversation query to provide an appropriate response.")
    ]

# @mcp.tool(
#     name="delete_current_context_data",
#     description="Delete all data within the 'contextual_data' collection that is not relevant to the current chat session to avoid bloating up the vector database with unneccessary information. It is best to use this tool if input_token starts exceeding max token count."
# )
# def delete_context_data():
    
    
@mcp.tool(
    name="count_claude_message_tokens",
    description="returns the total input token that is being used for the current query within the present chat session."
)
def count_claude_message_tokens(current_query : str) -> int:
    return anthropic.Anthropic().messages.count_tokens(
        model="claude-3-7-sonnet-20250219",
        messages=[
            {
                "role" : "user",
                "content" : current_query
            }
        ]
    ).json()["input_tokens"]
    
# @mcp.prompt(
#     name="track_database_history",
#     description="primary vector database that contains information is complete_collection"
# )
# def fetch_data(query : str) -> list[base.Message]:
#     return [
#         base.UserMessage(f"Based on the query {query}, reference the 'complete_collection' data for relevant context."),
#         base.AssistantMessage("I will reference the vector database collection named 'complete_collection' to retrieve contextually accurate response.")
#     ]
# TODO : refactor/remove if needed
# define list of relevant resources
# this resource is kind of useless
# @mcp.resource("context_history://{user_query}/{collection_name}/{relevant_docs}")
# def get_user_query_history(user_query : str, collection_name : str="contextual_data", relevant_docs : str = "5"):
#     '''
#     Utilizes the tool get_user_query_history, think of it as a wrapper around the tool and allows for propagating the parameters from resource to tool.
#     '''
#     result = retrieve_user_query_history(user_query, collection_name, int(relevant_docs))
#     print(f"contextual retrieval lookup result:\n\n\n{result}")
#     return result

# # TODO : implement as part of enhancement
# @mcp.resource("query_history://user_query")
# def retrieve_user_query():
#     pass

# TODO : delete later
# NOTE : for reference, delete later
# async def test_server_locally():
#     print("\n--- Testing Server Locally ---")
#     mcp_client = Client(mcp)

#     # Clients are asynchronous, therefore requires async context manager
#     # .call_tool method takes in name of tool as first param and parameter and value in the form of a dictionary
#     #
#     # .get_prompt method follows the same structure as .call_tool
#     async with mcp_client:
#         # call list of tools
#         echo_result = await mcp_client.call_tool("echo", {
#             "message" : "well hello there"
#         })
#         print(f"called echo tool : {echo_result}")

#         context_retrieval_tool_result = await mcp_client.call_tool("context_retriever", {
#             "user_query" : "who was abraham lincoln?",
#             "number_of_relevant_context" : 10
#         })
#         print(f"called context_retriever tool : {context_retrieval_tool_result}")

#         database_peek_result = await mcp_client.call_tool(
#             "peek_at_database", {
#                 "number_of_rows" : 3,
#                 "name_of_collection" : "complete_collection"
#             }
#         )
#         print(f"called database peek tool : {database_peek_result}")

#         existing_collection_name_modification_result = await mcp_client.call_tool(
#             "modify_collection_name", {
#                 "original_collection" : "complete_collection",
#                 "new_collection_name" : "modified_complete_collection"
#             }
#         )
#         print(f"called tool to modify collection name : {existing_collection_name_modification_result}")

#         get_collection_list_result = await mcp_client.call_tool("get_list_of_collections")
#         print(f"called tool to get list of available collections : {get_collection_list_result}")

#         # call prompts
#         prompt_message = await mcp_client.get_prompt("convert", {
#             "vector_data" : database_peek_result
#         })
#         print(f"prompt call result : {prompt_message}")

if __name__ == "__main__":
    mcp.run(transport='stdio')
    # asyncio.run(test_server_locally())
