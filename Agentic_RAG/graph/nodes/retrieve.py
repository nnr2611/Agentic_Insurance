from typing import Any, Dict
from graph.state import GraphState
from ingestion import retriever
#This retriever by now will reference the local vectorDB which contains the document chunks indexed - stored as embeddings

#Like i mentioned in one note - state(GraphState) - will always be the input for a Node
#SO it takes the State as input and returns a Dictionary - which says what to update in the state
def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---Retrieve----")
    #Extracting the question from the curernt state
    question = state['question']
    #Since the Langchain retriever is invoked with the question - it will use the question to do semantic search on the docs stored in the vectorDB - and fetch us relevant docs
    documents = retriever.invoke(question)
    #In the return statement we are trying to update the current state with the documents generated from retrieve
    return {"documents": documents, "question": question}