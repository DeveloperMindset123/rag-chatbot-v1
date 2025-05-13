# type : ignore
# TODO : integrate openAI agents sdk and google gemini model here to filter down context for for anthropic to use

from typing import Optional
from contextlib import AsyncExitStack
from chromaDB import client
from google import genai
from google.genai import types
import traceback
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datetime import datetime
import json
import os
import logging

from anthropic import Anthropic
from anthropic.types import Message


class MCPClient:
    def __init__(self):
        logger = logging.getLogger(__name__)
        logging.basicConfig(filename="mcp_client_log.log", level=30)

        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = (
            AsyncExitStack()
        )  # combines both synchronous and asynchronous context managers
        self.llm = Anthropic()
        self.tools = []
        self.messages = []
        self.info_logger = logger
        self.context_history_database = client.get_or_create_collection(
            name="contextual_data",
            metadata={
                "description": "chroma db vector database that stores relevant contexual information to keep track of user query."
            },
        )
        self.model_choice = "claude"

    async def set_model(self, model_choice: str):
        self.info_logger.info(f"Updated model to {model_choice}")
        self.model_choice = model_choice
        return f"Successfully set model to {model_choice}"

    async def get_model_choice(self) -> str:
        return self.model_choice

    # connect to the MCP server
    async def connect_to_server(self, server_script_path: str):
        try:
            if self.context_history_database.count() > 0:
                client.delete_collection(name="contextual_data")

            self.context_history_database = client.get_or_create_collection(
                name="contextual_data",
                metadata={
                    "description": "chroma db vector database that stores relevant contexual information to keep track of user and llm query."
                },
            )
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

            # self.logger.info("Connected to MCP server")
            self.info_logger.info("Connected to MCP server")

            mcp_tools = await self.get_mcp_tools()
            self.tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in mcp_tools
            ]

            self.info_logger.info(
                f"Available tools: {[tool['name'] for tool in self.tools]}"
            )

            return True

        except Exception as e:
            # self.logger.error(f"Error connecting to MCP server: {e}")
            self.info_logger.error(f"Error connecting to MCP server: {e}")
            traceback.print_exc()
            raise

    # get mcp tool list
    async def get_mcp_tools(self):
        try:
            response = await self.session.list_tools()
            return response.tools
        except Exception as e:
            self.info_logger.error(f"Error getting MCP tools: {e}")
            raise

    async def get_prompt_list(self):
        try:
            response = await self.session.list_prompts()
            return response.prompts
        except Exception as e:
            self.info_logger.error(f"Error getting MCP prompts: {e}")
            raise

    # process query
    async def process_query(self, query: str):
        try:
            self.info_logger.info(f"Processing query : {query}")
            user_message = {"role": "user", "content": query}
            self.messages = [user_message]

            while True:
                response = await self.call_llm()

                # the response is a text message
                if response.content[0].type == "text" and len(response.content) == 1:
                    assistant_message = {
                        "role": "assistant",
                        "content": response.content[0].text,
                    }
                    self.messages.append(assistant_message)
                    await self.log_conversation()
                    break

                # the response is a tool call
                assistant_message = {
                    "role": "assistant",
                    "content": response.to_dict()["content"],
                }
                self.messages.append(assistant_message)
                await self.log_conversation()

                for content in response.content:
                    if content.type == "tool_use":
                        tool_name = content.name
                        tool_args = content.input
                        tool_use_id = content.id
                        self.info_logger.info(
                            f"Calling tool {tool_name} with args {tool_args}"
                        )
                        try:
                            result = await self.session.call_tool(tool_name, tool_args)
                            self.messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": tool_use_id,
                                            "content": result.content,
                                        }
                                    ],
                                }
                            )
                            await self.log_conversation()
                        except Exception as e:
                            self.info_logger.error(
                                f"Error calling tool {tool_name}: {e}"
                            )
                            raise

            return self.messages

        except Exception as e:
            self.info_logger.error(f"Error processing query: {e}")
            raise

    # call llm
    async def call_llm(self):
        try:
            self.model_choice = await self.get_model_choice()
            print(f"retrieved choice of model : {self.model_choice}")
            match self.model_choice.lower().strip():
                case "claude":
                    self.info_logger.info("Calling Antrhopic")
                    print("Calling Antrhopic")
                    return self.llm.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=3500,
                        messages=self.messages,
                        tools=self.tools,
                    )
                case "gemini":
                    self.info_logger.info("Calling Gemini")
                    print("Calling Gemini")
                    self.llm = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                    self.tools = await self.get_mcp_tools()
                    gemini_response = self.llm.models.generate_content(
                        model="gemini-2.5-pro-exp-03-25",
                        contents=str(
                            self.messages[0]["content"]
                        ),  # needs to be a string
                        config=types.GenerateContentConfig(
                            temperature=0,
                            tools=[
                                types.Tool(
                                    function_declarations=[
                                        {
                                            "name": tool.name,
                                            "description": tool.description,
                                            "parameters": {
                                                k: v
                                                for k, v in tool.inputSchema.items()
                                                if k
                                                not in [
                                                    "additionalProperties",
                                                    "$schema",
                                                ]
                                            },
                                        }
                                    ]
                                )
                                for tool in self.tools
                            ],
                        ),
                    )
                    print(f"{self.messages[0]["content"]}")
                    print(
                        f"gemini response : {gemini_response.candidates[0].content.parts[0]}"
                    )
                    return gemini_response
                case _:
                    return f"error with call_llm() function."

        except Exception as e:
            self.info_logger.error(f"Error calling LLM: {e}")
            raise

    # cleanup
    async def cleanup(self):
        try:
            await self.exit_stack.aclose()
            self.info_logger.info("Disconnected from MCP server")
        except Exception as e:
            self.info_logger.error(f"Error during cleanup: {e}")
            traceback.print_exc()
            raise

    async def log_conversation(self):
        os.makedirs(
            "/Users/ayandas/Desktop/zed-proj/shield-takehome-proj/rag-chatbot-v1/rag-backend/conversations",
            exist_ok=True,
        )

        serializable_conversation = []

        for message in self.messages:
            try:
                serializable_message = {"role": message["role"], "content": []}

                # Handle both string and list content
                if isinstance(message["content"], str):
                    serializable_message["content"] = message["content"]
                elif isinstance(message["content"], list):
                    for content_item in message["content"]:
                        if hasattr(content_item, "to_dict"):
                            serializable_message["content"].append(
                                content_item.to_dict()
                            )
                        elif hasattr(content_item, "dict"):
                            serializable_message["content"].append(content_item.dict())
                        elif hasattr(content_item, "model_dump"):
                            serializable_message["content"].append(
                                content_item.model_dump()
                            )
                        else:
                            serializable_message["content"].append(content_item)

                serializable_conversation.append(serializable_message)
            except Exception as e:
                self.info_logger.error(f"Error processing message: {str(e)}")
                self.info_logger.debug(f"Message content: {message}")
                raise

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = os.path.join("conversations", f"conversation_{timestamp}.json")

        try:
            with open(filepath, "w") as f:
                json.dump(serializable_conversation, f, indent=2, default=str)
        except Exception as e:
            self.info_logger.error(f"Error writing conversation to file: {str(e)}")
            self.info_logger.debug(
                f"Serializable conversation: {serializable_conversation}"
            )
            raise
