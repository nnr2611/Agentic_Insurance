from typing import Any, Dict
from graph.chains.generation import generation_chain
from graph.state import GraphState

def generate(state: GraphState) -> Dict[str, Any]:
    print("-- GENERATE --")
    question = state['question']
    documents = state['documents']
    generation = generation_chain.invoke({"context":documents,"question":question})
    #If you look at the state file - till this point question, documents and the web_search fields were all populated (as web search node comes before generation node)
    #So for the generation node - this is the first time its getting populated
    return {"documents":documents,"question":question,"generation":generation}