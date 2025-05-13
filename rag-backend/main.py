"""
List of things to implement:
1. create an ephemeral client within mcp_client that will allow for storage and retrieval of user and assistnat responses (should be stored in the form of list of strings) and deleted after usage

2. create a new tool (replace the current contextual retrieval tool) that will be able to lookup and retrieve context history for reference

3. context history for user and assistant messages should be coming in from the frontend side in the form of a dictionary in the following format:

chat_history : { user_history : [], assistant_history : [] }
"""

from fastapi import FastAPI, HTTPException  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from pydantic import BaseModel  # type: ignore
from typing import Dict, Any, Union
from contextlib import asynccontextmanager
from mcp_client import MCPClient
from dotenv import load_dotenv  # type: ignore
from pydantic_settings import BaseSettings  # type: ignore
from agents import Agent, Runner

load_dotenv()


# TODO : Fix (find some workaround (potentially using sys))
class Settings(BaseSettings):
    server_script_path: str = (
        "/Users/ayandas/Desktop/zed-proj/shield-takehome-proj/rag-chatbot-v1/rag-backend/server.py"
    )


settings = Settings()

final_object_output = [{"title": "", "corresponding_points": [], "conclusion": ""}]


async def get_openAI_Agent_list():
    principal_software_engineer = Agent(
        name="Software Engineer",
        instructions=f"Convert markdown format data into appropriate JSON serializable data in the following format {final_object_output}. The final response should strictly adhere to the format I have provided. It should be array of object format containing the keys 'title', 'corresponding_points' and 'conclusion'",
    )

    professor = Agent(
        name="Professor",
        instructions="You are a helpful assistant who can take raw string data and convert it into easily readable markdown format.",
    )

    return [professor, principal_software_engineer]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    instantiates MCP Client and attempts to use the client to establish a connection with the server

    async context manager allows for execution of code prior to yield line.
    """
    client = MCPClient()
    try:
        connected = await client.connect_to_server(settings.server_script_path)
        if not connected:
            raise HTTPException(
                status_code=500, detail="Failed to connect to MCP server"
            )

        """
        state : attribute found within FastAPI class. A state object for the application that doesn't change from request to rqeuest.
        
        This attribute is inherited from Starlette.
        
        This allows us to inherit all the methods within MCPClient (alongside the built in ones and access them).
        """
        app.state.client = client
        yield
    except Exception as e:
        print(f"Error during lifespan: {e}")
        raise HTTPException(status_code=500, detail="Error during lifespan") from e
    finally:
        # shutdown
        await client.cleanup()


app = FastAPI(title="MCP Client API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class QueryRequest(BaseModel):
    query: str


class Message(BaseModel):
    role: str
    content: Any


class ToolCall(BaseModel):
    name: Any
    args: Dict[str, Any]


class FinalObjectOutput(BaseModel):
    title: str
    corresponding_points: list[str]
    conclusion: str


# tells frontend server is working
@app.get("/ping")
async def server_check():
    return True


@app.post("/query")
async def process_query(request: QueryRequest):
    import json

    """Process a query and return the response"""
    try:
        # await app.state.client.set_model("Gemini")        # TODO : implement support, claude works for now
        messages = await app.state.client.process_query(request.query)
        # print(f"final_message : {messages}")

        agent_list = await get_openAI_Agent_list()
        # json_serialized_data = await Runner.run(agent_list[0], input=str(messages))
        nlp_response = await Runner.run(agent_list[0], input=str(messages))
        # json_serialized_data = await Runner.run(
        #     agent_list[1], input=str(nlp_response.final_output)
        # )
        # json_formatted_data = json.dumps(
        #     json.loads(json_serialized_data.final_output), indent=2
        # )
        # print(json_serialized_data.final_output)
        print(nlp_response.final_output)
        # print(f"{messages}")
        return {"final_response": nlp_response.final_output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools")
async def get_tools():
    """Get the list of available tools"""
    try:
        tools = await app.state.client.get_mcp_tools()
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in tools
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/prompts")
async def get_prompts():
    """
    get list of available prompts
    """
    try:
        prompts = await app.state.client.get_prompt_list()
        return {
            "prompts": [
                {
                    "name": prompt.name,
                    "description": prompt.description,
                }
                for prompt in prompts
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn  # type: ignore

    uvicorn.run(app, host="0.0.0.0", port=8000)
