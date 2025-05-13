# type : ignore

# This is mcp client for CLI communication.
"""
Implementation checklist: (delete after)
TODO : First figure out if hashmap can be used to store context (and using a seperate LLM to make the current contextual information JSON serializable.) --> NOTE that you can either use Ollama or OpenAI for this (whichever is least expensive, have ollama as primary and openAI for fallback, to save costs)

TODO : see if you can store the context within a new directory of chromaDB vector (will requier another chromaDB instance) --> this will require writting additional tools/resources to expose to the LLM for use

TODO : Implement the gemini endpoint as alternative for users to choose between anthropic and gemini for their particular LLM.
"""

import asyncio
import json
import chromaDB
import random
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datetime import datetime
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        # self.message_context = []       # NOTE : must be array of objects

        self.context_history_database = chromaDB.client.get_or_create_collection(
            name="contextual_data",
            metadata={
                "description": "chroma db vector database that stores relevant contextual information to keep track of user query"
            },
        )

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        message_context: list[any] = [{"role": "user", "content": query}]

        response = await self.session.list_tools()
        available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]

        # print(f"current message context : {json.dumps(message_context)}")
        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=3000,
            messages=message_context,
            tools=available_tools,
        )

        # Process response and handle tool calls
        tool_results = []
        final_text, assistant_message_content = [], []

        for content in response.content:
            if content.type == "text":
                final_text.append(content.text)
                assistant_message_content.append(content)
            elif content.type == "tool_use":
                tool_name = content.name
                tool_args = content.input

                # Execute tool call (built-in mcp method)
                result = await self.session.call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                assistant_message_content.append(content)
                message_context.append(
                    {"role": "assistant", "content": assistant_message_content}
                )

                message_context.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result.content,
                            }
                        ],
                    }
                )

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2000,
                    messages=message_context,
                )

                final_text.append(response.content[0].text)

        self.message_context = message_context
        print(f"length of message context : {len(str(message_context))}")
        print("started adding contextual history to chroma db")
        # NOTE : might fail
        #
        metadatas = []
        metadatas.append(
            {
                "identificaton": "user query history",
                "user": "ayan das",
                "created_at": str(datetime.now()),
            }
        )
        self.context_history_database.add(
            documents=str(message_context),
            metadatas=metadatas,
            # ids=random.sample(range(1,len(str(message_context)) * 2), len(str(message_context)))
            ids=str(random.randint(0, 1000000)),
        )
        print("finished pushing data to chroma db")
        print(f"content within message array : {message_context}")
        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    self.context_history_database.delete_collection(
                        name="contextual_data"
                    )
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

    def toJson(self, data):
        """converts any data to make it json serailzable"""
        return json.dumps(data, default=lambda o: o.__dict__, sort_keys=True, indent=4)


async def main():
    server_script_path = ""
    if len(sys.argv) < 2:
        # print("Usage: python client.py <path_to_server_script>")
        # sys.exit(1)
        server_script_path = "/Users/ayandas/Desktop/zed-proj/shield-takehome-proj/rag-chatbot-v1/rag-backend/server.py"

    client = MCPClient()
    try:
        await client.connect_to_server(
            server_script_path if len(sys.argv) < 2 else sys.argv[1]
        )
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys

    asyncio.run(main())
