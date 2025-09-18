from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from database.rag_search import semantic_search, generate_answer

app = FastAPI()

# --- Request Models ---
class SearchRequest(BaseModel):
    query_text: str
    top_k: int = 5

class AnswerRequest(BaseModel):
    query_text: str
    retrieved_chunks: List[Dict]

class RagRequest(BaseModel):
    query_text: str
    top_k: int = 5

# --- Endpoints ---
@app.post("/search")
def search(req: SearchRequest):
    results = semantic_search("markdown_chunks", req.query_text, req.top_k)
    return {"results": results}

@app.post("/answer")
def answer(req: AnswerRequest):
    answer = generate_answer(req.query_text, req.retrieved_chunks)
    return {"answer": answer}

@app.post("/rag")  # optional combined endpoint
def rag(req: RagRequest):
    results = semantic_search("markdown_chunks", req.query_text, req.top_k)
    answer = generate_answer(req.query_text, results)
    return {"results": results, "answer": answer}
