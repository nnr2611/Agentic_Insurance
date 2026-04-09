from typing import List, TypedDict

#We need to have question - for reference - whether to see that the documents retrieved are relevant to the question or not
#Save the documents that help us answer the questions as well - documents that we get back from search results
#Boolean - web_search - whether we need to do extra search or not
#generation field - has the generated answer from the LLM
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """
    question: str
    generation: str
    web_search: bool
    documents: List[str]