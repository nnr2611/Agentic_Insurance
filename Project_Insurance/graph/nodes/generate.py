# generate.py
from typing import Any, Dict
from graph.chains.generation import generation_chain
from graph.state import ClaimGraphState

def generate(state: ClaimGraphState) -> Dict[str, Any]:
    print("-- GENERATE --")
    
    question = state['question']
    chunk_docs = state['chunk_docs']

    extracted_fields = generation_chain.invoke({
        "context": chunk_docs,
        "question": question
    })

    general_fields = state['general_fields'].copy()
    general_fields["Extracted Fields"] = extracted_fields

    return {
        "general_fields": general_fields,
        "question": question,
        "chunk_docs": chunk_docs
    }
