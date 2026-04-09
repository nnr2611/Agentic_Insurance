from typing import Any, Dict
from graph.state import ClaimGraphState
from ingestion import retriever

def retrieve(state: ClaimGraphState) -> Dict[str, Any]:
    print("---Retrieve----")
    question = state['question']
    chunk_docs = retriever.invoke(question)
    return {"chunk_docs": chunk_docs, "question": question}