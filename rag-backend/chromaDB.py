# type : ignore
# run this script to store data within chroma db
from typing import Any
from datasets import load_dataset
import chromadb
from datetime import datetime
from chromadb.config import Settings

client = chromadb.HttpClient(host="localhost", port=9000)       # recommended
local_client = chromadb.PersistentClient(path="/Users/ayandas/Desktop/zed-proj/shield-takehome-proj/rag-chatbot-v1/rag-backend/chromaDbData")

# Constants
HUGGINGFACE_DATASET_API = "rag-datasets/rag-mini-wikipedia"
HUGGINGFACE_LOAD_DATASET_2ND_PARAM = "question-answer"

class ChromaDBVectorDatabase:
    def __init__(self, collection_name : str ="complete_collection", client_instance:Any=None):
        # Initialize ChromaDB client and create a collection
        self.client = client_instance
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "chroma db vector collection",
                "created": str(datetime.now())
            }
        )
        self.collection_name=collection_name

    def get_collection_list(self):
        return self.client.list_collections()

    def create_new_collection(self,collection_name : str):
        self.client.create_colletion(name=collection_name,
            metadata={
            "author" : "Ayan Das",
            "description" : "chroma db vector collection",
            "created" : str(datetime.now())
        })

    def store_data(self, data):
        """
        Store a list of dictionaries in the ChromaDB collection.
        Each dictionary should have 'question', 'answer', and 'id'.
        """
        # Extract the test data
        test_data = list(data['test'])

        # Prepare lists for batch insertion
        documents = []
        metadatas = []
        ids = []

        # Process each entry
        for entry in test_data:
            # Combine question and answer for the document
            document_text = f"Question: {entry['question']} Answer: {entry['answer']}"
            documents.append(document_text)

            # Create metadata
            metadata = {"author": "ayan das", "question": entry['question']}
            metadatas.append(metadata)

            # Convert ID to string to ensure compatibility
            ids.append(str(entry['id']))

        # Print some debug information
        print(f"Number of documents: {len(documents)}")
        print(f"Number of metadatas: {len(metadatas)}")
        print(f"Number of ids: {len(ids)}")
        print(f"Sample documents: {documents[:5]}")
        print(f"Sample metadatas: {metadatas[:5]}")
        print(f"Sample ids: {ids[:5]}")

        # Add documents in batches to avoid potential size limits
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            self.collection.add(
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
            print(f"Added batch {i//batch_size + 1} ({i} to {batch_end})")

    def search(self, query, n_results=5):
        """
        Search for the most relevant documents based on the query.
        Returns the top n_results matching documents.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
    def deleteCollection(self, collection_to_delete : str):
        return self.client.delete_collection(name=collection_to_delete)

# @mcp.tool()
#
# this is to experiment and check if the collection query is working as intended --> only used for local testing
def get_huggingface_data() -> dict[str, Any] | None:
    """Make a request to the huggingface dataset api to retrieve the data and send it to chromaDB vector database"""

    try:
        return {
            "status" : "success",
            "status_code" : 200,
            "message" : "data loaded successfully",
            "data" : load_dataset(HUGGINGFACE_DATASET_API, HUGGINGFACE_LOAD_DATASET_2ND_PARAM)
        }

        # not needed architecturally, use for reference purpose only.
        # collection_name = "complete_collection"
        # collection = ChromaDBVectorDatabase(collection_name)

        # collection.store_data(data)
        # sample_query = "was abraham lincoln the first president of the united states?"
        # search_results = collection.search(sample_query)

        # print(f"retrieved search results : \n {search_results}")

        # return {"status": "success", "message": "Data loaded and stored successfully", "collection" : collection }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "status": "error",
            "status_code" : 503,
            "message": str(e)}

if __name__ == "__main__":
    huggingface_data = get_huggingface_data()
    chroma_instance = ChromaDBVectorDatabase("complete_collection", client)
    chroma_instance.store_data(huggingface_data["data"])
