from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import END, StateGraph
from graph.graph_constants import STORE, RETRIEVE, GENERATE, GRADE_DOCUMENTS, ENSEMBLE_MODEL
from graph.nodes.store import store
from graph.nodes.retrieve import retrieve
from graph.nodes.generate import generate
from graph.nodes.ensemble_model import ensemble_model
from graph.state import ClaimGraphState

from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.retrieval_grader import retrieval_grader

workflow = StateGraph(ClaimGraphState)

#Creating conditional edges from generation Node to the Hallucination checker chain
def grade_generation_grounded_in_docs_and_question(state: ClaimGraphState) -> str:
    print("--- ChECK HALLUCINATIONS ---")
    original_docs = state['original_docs']
    chunk_docs = state['chunk_docs']
    question = state['question']
    score = hallucination_grader.invoke(
        {"original_docs":original_docs,"chunk_docs":chunk_docs}
    )
    if hallucination_grade := score.binary_score:
        print("--- DECISION: GENERATION IS GROUNDED IN THE DOC ---")
        print("--- RETRIEVED CHUNK DOCS vs QUESTION (FINAL CHECK) ---")
        score = retrieval_grader.invoke({"question":question,"chunk_docs":chunk_docs})
        if answer_grade := score.binary_score:
            print("--- DECISION: GENERATION ADDRESSES QUESTION ---")
            return "useful"
        else:
            print("--- DECISION: GENERATION DOES NOT ADDRESS THE QUESTION---")
            return "not useful"
    else:
        print("--- DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, REGENERATE AGAIN FROM DOCUMENTS ---")
        return "not supported"

#Adding Nodes
workflow.add_node(STORE, store)
workflow.add_node(RETRIEVE, retrieve)
# workflow.add_node(GRADE_DOCUMENTS,grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(ENSEMBLE_MODEL, ensemble_model)

#Adding Edges and entry points
workflow.set_entry_point(STORE)
workflow.add_edge(STORE, RETRIEVE)
workflow.add_edge(RETRIEVE, GENERATE)
# workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_docs_and_question,
    {
        "not supported": GENERATE,
        "useful": ENSEMBLE_MODEL,
        "not useful": RETRIEVE,
    },
)
# workflow.add_edge(GENERATE,ENSEMBLE_MODEL)
workflow.add_edge(ENSEMBLE_MODEL,END)

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")