# type : ignore
import asyncio
from typing import Optional, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

# import openai
#

load_dotenv()

class MCPClient:
    def __init__(self):
        self.session : Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_server(self, server_script_path : str):
        """connect to an MCP server

        Args:
            server_script_path : Path to the server script (.py or .js)
        """
        is_python : bool = server_script_path.endswith(".py")
        is_javascript : bool = server_script_path.endswith(".js")

        if not (is_python or is_javascript):
            raise ValueError("Server script must end with a .py or .js")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # list available tools
        response = await self.session.list_tools()
        # print(f"response for tools list : {response}")
        tools = response.tools()
        print("\n connected to server with tools:", [tool.name for tool in tools])

        async def process_query(self, query : str) -> str:
            """
            process query using claude or openAI and available tools and prompts
            """
            selected_ai_agent = input("do you want to use anthropic or openai?")
            messages = [{
                "role" : "user",
                "content" : query
            }]

            # NOTE : attempt at retrieving and adding prompt
            context_retrieval_tool_result = await self.session.call_tool("contex_retriever", {
                "user_query" : query,
                "number_of_relevant_context" : 10
            })
            mcp_server_prompt = await self.session.get_prompt("convert", {
                "vector_data" : str(context_retrieval_tool_result)
            })
            print(f"retrieved mcp server prompt data : {mcp_server_prompt}")
            messages.append(mcp_server_prompt)

            response = await self.session.list_tools()
            available_tools = [{
                "name" : tool.name,
                "description" : tool.description,
                "input_schema" : tool.inputSchema

            } for tool in response.tools()]

            print(f"corresponding available tools are {available_tools}")

            # initiate claude (1st of 2 invokation)
            final_text, assistant_message_content = [], []
            response = self.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=messages,
                tools=available_tools
            )

            for content in response.content:
                if content.type == 'text':
                    final_text.append(content.text)
                    assistant_message_content.append(content)

                elif content.type == 'tool_use':
                    tool_name = content.name
                    tool_args = content.input

                    # execute tool call
                    result = await self.session.call_tool(tool_name, tool_args)
                    print(f"tool call result : {result}")

                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}")

                    assistant_message_content.append(content)
                    messages.append({
                        "role" : "assistant",
                        "content" : assistant_message_content
                    })
                    tool_data : Any = {
                                    "role": "user",
                                    "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": content.id,
                                        "content": result.content
                                    }
                                ]
                            }
                    messages.append(tool_data)

                    # get next reponse from claude (2 of 2 invokation)
                    response2 = self.antrhopic.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=1000,
                        messages=messages,
                        tools=available_tools
                    )
                    print(f"retrieved response from second call: \n {response2}")
                    final_text.append(response.content[0].text)
            return "\n".join(final_text)

        async def cleanup_data(self):
            await self.exit_stack.aclose()

        async def chat_loop(self):
            '''Run an interactive chat loop'''
            print("\n Client started!")
            print("Type your queries or 'quit' to exit.")

            while True:
                try:
                    query = input("\nQuery: ").strip()

                    if query.lower() == 'quit':
                        break

                    response = await self.process_query(query)

                except Exception as e:
                    print(f"\n Error with interactive chat : {e}")

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
