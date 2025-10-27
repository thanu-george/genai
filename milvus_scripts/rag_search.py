from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import subprocess
from typing import List, Dict

# --- Ollama wrapper ---
def ollama_generate(prompt: str, model: str = "llama3:8b") -> str:
    cmd = ["ollama", "run", model, prompt]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

# --- Semantic Search ---
def semantic_search(collection_name: str, query_text: str, top_k: int) -> List[Dict]:
    # Connect to Milvus
    connections.connect("default", host="localhost", port="19530")

    # Load collection
    collection = Collection(collection_name)
    collection.load()

    # Create query embedding
    embedder = SentenceTransformer("all-mpnet-base-v2")
    query_vector = embedder.encode(query_text).tolist()

    # Perform vector search
    results = collection.search(
        data=[query_vector],
        anns_field="chunk_text_vector",
        param={"metric_type": "COSINE", "params": {"ef": 256}},
        limit=top_k,
        output_fields=["document_id", "chunk_order", "chunk_text"]
    )

    # Return chunks as dicts
    hits = sorted(results[0], key=lambda x: x.distance)  # sort ascending (smaller = better)

    return [
    {
        "document_id": hit.entity.get("document_id"),
        "score": hit.distance,
        "chunk_order": hit.entity.get("chunk_order"),
        "chunk_text": hit.entity.get("chunk_text"),
    }
    for hit in hits
]

# --- Answer Generation ---
def generate_answer(query_text: str, retrieved_chunks: List[Dict], model_name: str = "llama3:8b") -> str:
    context = "\n\n---\n\n".join([chunk["chunk_text"] for chunk in retrieved_chunks])
    prompt = f"""You are a helpful and precise assistant. Give a concise and brief answer.
Use only the following context to answer the question. If the answer cannot be found in the context, say "The context does not provide enough information."

Context:
{context}

Question: {query_text}

Answer:"""
    return ollama_generate(prompt, model=model_name)

# --- For CLI testing ---
if __name__ == "__main__":
    query = "What is tax exemption under section 80IAC?"
    chunks = semantic_search("markdown_chunks", query, top_k=5)

    print("\n=== Retrieved Chunks ===")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nResult {i} (distance={chunk['score']:.4f})")
        print("Document:", chunk["document_id"])
        print("Text:", chunk["chunk_text"])

    answer = generate_answer(query, chunks)
    print("\n=== Final Answer ===")
    print(answer)
