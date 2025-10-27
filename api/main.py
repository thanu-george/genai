'''from fastapi import FastAPI
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
    return {"results": results, "answer": answer}'''

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional

from database.rag_search import semantic_search, generate_answer
from session_manager import SessionManager
from retrieve.config.retrieval_config import RetrievalConfig

# --- Setup ---
cfg = RetrievalConfig()   # must define redis_host, redis_port, redis_db, session_ttl_seconds
session_mgr = SessionManager(cfg)
app = FastAPI()

# --- Models ---
class QueryRequest(BaseModel):
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    query: str
    new_chat: bool = False
    top_k: int = 5

# --- Endpoints ---
@app.post("/query")
def query(req: QueryRequest):
    # assign user_id if new
    if not req.user_id:
        req.user_id = session_mgr.create_user()

    # create new session if needed
    if req.new_chat or not req.session_id:
        req.session_id = session_mgr.create_session(req.user_id)

    # log user query
    session_mgr.append_turn(req.session_id, "user", req.query)

    # retrieve & answer
    results = semantic_search("markdown_chunks", req.query, req.top_k)
    answer = generate_answer(req.query, results)

    # log assistant answer
    session_mgr.append_turn(req.session_id, "assistant", answer)

    return {
        "user_id": req.user_id,
        "session_id": req.session_id,
        "answer": answer,
        "history": session_mgr.get_history(req.session_id),
    }

@app.get("/sessions/{user_id}")
def list_sessions(user_id: str):
    return {"sessions": session_mgr.get_user_sessions(user_id)}

@app.get("/history/{session_id}")
def get_history(session_id: str):
    return {"history": session_mgr.get_history(session_id)}



