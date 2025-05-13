# # type : ignore
"""'
Python script to test if client server interaction is working as intended.
not the official mcp-client, refer to mcp-client.py for primary client implementation

NOTE : It is best to run this script if you just want to check and ensure that all the relevant tools are working as intended.
"""

from fastmcp import Client
import asyncio


async def interact_with_server():
    print("---creating client---")
    mcp_client = Client(
        "/Users/ayandas/Desktop/zed-proj/shield-takehome-proj/rag-chatbot-v1/rag-backend/server.py"
    )

    # print(f"Client configured to connect to : {mcp_client.target}")
    try:
        async with mcp_client:
            print("---Client Connected---")

            # call all relevant tools

            # This is a test tool
            echo_result = await mcp_client.call_tool(
                "echo", {"message": "well hello there"}
            )
            print(f"called echo tool : {echo_result}")

            context_retrieval_tool_result = await mcp_client.call_tool(
                "context_retriever",
                {
                    "user_query": "who was abraham lincoln?",
                    "number_of_relevant_context": 10,
                },
            )
            print(f"called context_retriever tool : {context_retrieval_tool_result}")

            database_peek_result = await mcp_client.call_tool(
                "peek_at_database",
                {"number_of_rows": 3, "name_of_collection": "complete_collection"},
            )
            print(f"called database peek tool : {database_peek_result}")

            existing_collection_name_modification_result = await mcp_client.call_tool(
                "modify_collection_name",
                {
                    "original_collection": "complete_collection",
                    "new_collection_name": "modified_complete_collection",
                },
            )
            print(
                f"called tool to modify collection name : {existing_collection_name_modification_result}"
            )

            get_collection_list_result = await mcp_client.call_tool(
                "get_list_of_collections"
            )
            print(
                f"called tool to get list of available collections : {get_collection_list_result}"
            )

            # call prompts
            prompt_message = await mcp_client.get_prompt(
                "convert", {"vector_data": database_peek_result}
            )
            print(f"prompt call result : {prompt_message}")
            print(f"client is runnign on {mcp_client.settings.port}")
    except Exception as e:
        return f"Client failure due to {e}"


if __name__ == "__main__":
    asyncio.run(interact_with_server())
