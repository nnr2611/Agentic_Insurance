from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from constants import SYSTEM_PROMPT_ANS_GRADER

class GradeRetrieval(BaseModel):
    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

llm = ChatOpenAI(temperature=0)
structured_llm_grader = llm.with_structured_output(GradeRetrieval)

retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",SYSTEM_PROMPT_ANS_GRADER),
        ("human","User question: \n\n {question} \n\n LLM generation: {chunk_docs}"),
    ]
)

retrieval_grader: RunnableSequence = retrieval_prompt | structured_llm_grader













