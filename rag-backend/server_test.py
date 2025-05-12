import requests
import json
from chromaDB import ChromaDBVectorDatabase, client

# use for testing different scripts
BASE_URL = "http://127.0.0.1:6274"

def safe_get(url):
    try:
        res = requests.get(url)
        res.raise_for_status()
        try:
            return res.json()
        except Exception:
            return {"raw_response": res.text.strip(), "error": "Non-JSON response"}
    except Exception as e:
        return {"error": str(e)}

def safe_post(url, payload):
    try:
        res = requests.post(url, json=payload)
        res.raise_for_status()
        try:
            return res.json()
        except Exception:
            return {"raw_response": res.text.strip(), "error": "Non-JSON response"}
    except Exception as e:
        return {"error": str(e)}

def discover_mcp():
    print(f"🔍 Connecting to MCP Inspector at {BASE_URL}")

    # Try both endpoints
    for endpoint in ["/tools/list", "/api/tools/list"]:
        print(f"\n📦 Checking tools at {endpoint}:")
        tools = safe_get(f"{BASE_URL}{endpoint}")
        print(json.dumps(tools, indent=2))
        if "tools" in tools:
            break  # We found valid data

    if "tools" not in tools:
        print("\n⚠️ No tools found or MCP Inspector not serving expected JSON.")
        return

    for tool in tools["tools"]:
        print(f"\n⚙️ Testing tool: {tool['name']}")
        dummy_payload = {
            "args": {
                k: "test" if v == "str" else 1 for k, v in tool.get("parameters", {}).items()
            }
        }
        result = safe_post(f"{BASE_URL}/tools/{tool['name']}", dummy_payload)
        print(json.dumps(result, indent=2))

    print("\n📚 Checking greeting resource:")
    resource = safe_get(f"{BASE_URL}/resources/greeting://Alice")
    print(json.dumps(resource, indent=2))

if __name__ == "__main__":
    # discover_mcp()
    # chroma_instance = ChromaDBVectorDatabase("complete_collection", client)
    # search_res = chroma_instance.search("Was abraham lincoln a president?")
    # print(search_res["documents"])
    client.create_collection(name="experimental")
    collection_names = client.list_collections()
    # print(f"collection names before deletion {collection_names}")
    client.delete_collection(name="experimental")
    collection_names_modified = client.list_collections()
    # print(f"collections names after deletion {collection_names_modified}")
    complete_collection_instance = client.get_collection("complete_collection")

    # example of retrieving the top 10 vector data
    print(complete_collection_instance.peek(limit=10)["metadata"])
