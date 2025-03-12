"""
Description: indexer.py
- Uses the HF embedding model to encode the query
- Retrieves top-k semantically similar results in Milvus based on encoded query
"""

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

milvus_client = MilvusClient(uri="././milvus_data/2025_budget.db"
                             )
collection_name = "sg_budget"

# Check if the collection exists
if not milvus_client.has_collection(collection_name):
    print(f"Collection '{collection_name}' does not exist. Please run the indexer first.")
    exit(1)

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5"
                                      )
def emb_text(text):
    return embedding_model.encode([text], 
                                  normalize_embeddings=True).tolist()[0]

if __name__ == "__main__":
    question = "what is singapore's economy growth?"
    topk = 3
    # Search parameters
    search_params = {"metric_type": "IP", 
                     "params": {"nprobe": 10}}
    
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[emb_text(question)],  # Use the emb_text function to convert the question to an embedding vector
        limit=topk,  # Return top k results
        search_params=search_params,
        output_fields=["text"],  # Return the text field
        )
    
    # Handle empty results
    if not search_res or not search_res[0]:
        print("No relevant results found for the query.")
        exit(1)

    print("\n**Top Retrieved Documents for Query:**", question)

    for i, result in enumerate(search_res[0]):  
        score = result.get("distance", None)  
        text = result.get("entity", {}).get("text", "No text available")  

        print(f"\n**Result {i+1}:**")
        print(f"**Relevance Score:** {score:.3f}")
        print(f"**Extracted Text:** {text[:500]}...")  # Limit to 500 chars
        print("-" * 80)