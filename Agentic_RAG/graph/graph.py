from dotenv import load_dotenv
load_dotenv()

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader

from langgraph.graph import END, StateGraph
from graph.consts import RETRIEVE,GRADE_DOCUMENTS,GENERATE,WEBSEARCH
from graph.nodes import generate, retrieve, web_search
from graph.nodes.grade_documents import grade_documents
from graph.state import GraphState

#If you look at the generation graph in one Note - the grade docs node is going to web search or directly to generate - so the below function is to decide in which direction for it to go
def decide_to_generate(state):
    print("--- ASSESS GRADED DOCUMENTS ---")
    if state['web_search']:
        print("--- NOT ALL RETRIEVED DOCS ARE RELEVANT TO THE QUESTION ---")
        return WEBSEARCH
    else:
        print("--- DECISION GENERATE ---")
        return GENERATE

#Creating conditional edges from generation Node to the Hallucination checker chain
#The input is state - the state contains details - until the generation Node's output and this function output - will be the conditional edge deciding the next State 
def grade_generation_grounded_in_docs_and_question(state: GraphState) -> str:
    print("--- ChECK HALLUCINATIONS ---")
    question = state['question']
    documents = state['documents']
    generation = state['generation']

    #the input from the state contains documents fetched and the response given by generation node
    #The response that this grader chain gives is - a binar yscore stating whether this contains hallucinations or not
    score = hallucination_grader.invoke(
        {"documents":documents,"generation":generation}
    )

    #If this value is true-  there is no halucination - so positive grade
    if hallucination_grade := score.binary_score:
        print("--- DECISION: GENERATION IS GROUNDED IN THE DOC ---")
        print("--- GRADE GENERATION vs QUESTION (FINAL CHECK) ---")
        score = answer_grader.invoke({"question":question,"generation":generation})
        if answer_grade := score.binary_score:
            print("--- DECISION: GENERATION ADDRESSES QUESTION ---")
            return "useful"
        else:
            #This means that the answer obtained - is indeed grounded in the docs but its not useful as its not relevant to the question
            #in that case we will be doing a web based search
            print("--- DECISION: GENERATION DOES NOT ADDRESS THE QUESTION---")
            return "not useful"
    else:
        #If answer is not even grounded in docs
        #We will connect this back to generate - and regenrate until the generation is grounded to the docs
        print("--- DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, REGENERATE AGAIN FROM DOCUMENTS ---")
        return "not supported"



#So like we know state is the only entity that is constantly flowing with data acoss multiple nodes in the graph
#We need a common tempalte for that state - which we defined in GraphState
workflow = StateGraph(GraphState)

#Adding Nodes
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS,grade_documents)
workflow.add_node(WEBSEARCH,web_search)
workflow.add_node(GENERATE, generate)

#Adding Edges and entry points
workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS,decide_to_generate,{WEBSEARCH: WEBSEARCH, GENERATE: GENERATE})
workflow.add_edge(WEBSEARCH,GENERATE)
workflow.add_edge(GENERATE,END)

#Self RAG
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_docs_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)


app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")