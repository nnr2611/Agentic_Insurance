# store.py
from typing import Any, Dict
from graph.state import ClaimGraphState
import ingestion
from langchain_community.document_loaders import PyPDFLoader

def store(state: ClaimGraphState) -> Dict[str, Any]:
    print("---Store---")
    
    doc_path = state['doc_path']
    loader = PyPDFLoader(doc_path)
    docs_list = loader.load()

    full_text = "\n\n".join(doc.page_content for doc in docs_list)
    ingestion.ingest_text_to_vectordb(full_text)

    return {"doc_path": doc_path, "original_docs": full_text}
