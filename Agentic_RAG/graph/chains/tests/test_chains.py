from dotenv import load_dotenv
load_dotenv()

from pprint import pprint

from ..retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import hallucination_grader, GraderHallucinations
from ingestion import retriever

#This flwo is for - The retrieved documents turn out to be relevant to the question asked
def test_retrieval_grader_answer_yes() -> None:
    question = "LLM"
    docs = retriever.invoke(question)
    print(docs)
    #We are going to take the first document that we find and take its content
    #This means this doc will receive highest score
    #Since the question is on agent memory, the retrieved doc also should be on agent memory
    doc_text = docs[0].page_content
    #We are going to invoke the chain here - we are going to get back results which is going to be an object of GradeDocuments
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_text}
    )
    #This is the final check - that makes this testing relevant
    #The above document grader node - will give the output - res as obj f Graddocuments that contains binary score
    #Tha binary score we are checking now here as yes or no
    assert res.binary_score =="yes"

#Similar test condition for No binary score
def test_retrival_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizaa", "document": doc_txt}
    )

    assert res.binary_score == "no"

def test_generation_chain() -> None:
    #Since this test is not going to give output like the previous two tests - which have output as either yes or no - This will give a paragraph response
    #Therefore we are not adding any checkers - we are just running sanity checks on this
    question ="agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context":docs,"question":question})
    print(generation)
    pprint(generation)

def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question":question})
    #The answer from generation should be grounded in doc - if not -> hallucination
    res: GraderHallucinations = hallucination_grader.invoke(
        {"documents":docs, "generation":generation}
    )
    assert res.binary_score

def test_hallucination_grader_answer_no() -> None:
    question = "james bond"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question":question})
    #The answer from generation should be grounded in doc - if not -> hallucination
    res: GraderHallucinations = hallucination_grader.invoke(
        {"documents":docs, "generation":generation}
    )
    assert not res.binary_score









