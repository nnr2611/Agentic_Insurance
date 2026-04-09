from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

#Thsi class - -GradeDocuments is a pydantic model
#It will have a single field of binary score - which is a string -yes or no
class GradeDocuments(BaseModel):
    #This pydantic output format structuring is important because the LLM is going to utilise this structure dsescription - to decide whether this document is relevant or not (That is why we want hte score to be just yes or no)
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes or no'"
    )

#What LLM is going to do under teh hood is that - it will leverage function calling for every LLM call we make - we return a pydantic object and LLM is going to reutrn it in the schema we want - THAT IS SHOWN ABOVE
#GPT models (latest support function calling - sometimes other open source models might not support it)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system="""
You are a grade assessing relevance of a retrieved document to a user question. \n
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant
Give a binary score 'yes' or 'no' score to indicuate whether the document is relevant to the question
"""

#Now we will use chatprompt template from messages method
#plugin system message
# we will also put human message - as though showing those retrieved documents and the user question - saying like a human is asking the LLM whether this doc is relevant or not
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human","Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

#Finally we are creating the chain - retrieval grader - its taking the grader prompt and its going to pipe it with the LLM with the structured output request

retrieval_grader = grade_prompt | structured_llm_grader

