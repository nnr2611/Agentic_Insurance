#In this file we  are writing the Node implementation of - Grading all the docs to decide whether wawe want to filter them out or keep them in
#We will write a function that receives a state and in that state we will have question, fetch documents and our grader chain will decide whether each document is relevant or not if its not relvant -we will fitler it out
#And finally if we find any document which is not relevant - we will make the web searching flag as True - so as to search the web for further ans

from typing import Any, Dict
from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the recieved documents are relvant to the question
    if any document is not relevant, we will set a flag to run web search
    
    Args:
        state (dict): the current graph state
    
    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    
    """

    #Checking relevance of the documents - 
    print("--- CHECK DOCUMENT RELEVANCE TO QUESTION ---")
    question = state['question']
    documents = state["documents"]

    #initializing the output before processing
    filtered_docs=[]
    web_search=False

    for docu in documents:
        #THis output - score is the binary (yes or no) stating whether that docu contains relevancy or not
        score = retrieval_grader.invoke(
            {"question":question,"document":docu.page_content}
        )
        grade = score.binary_score
        if grade.lower() =="yes":
            print("--- GRADE: DOCUMENT RELEVANT ---")
            filtered_docs.append(docu)
        else:
            print("--- GRADE: DOCUMENT NOT RELEVANT ---")
            #Since if we get even one irrelevant document we are going to set web search as true only
            web_search=True
            continue
    return {"documents":filtered_docs,"question":question,"web_search":web_search}
    










