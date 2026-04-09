from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from constants import SYSTEM_PROMPT_HALLUCINATION

llm = ChatOpenAI(temperature=0)

class GraderHallucinations(BaseModel):
    "Binary score for hallucination present in the generation answer."
    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GraderHallucinations)

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",SYSTEM_PROMPT_HALLUCINATION),
        ("human","Original Documents: \n\n {original_docs} \n\n LLM generation: {chunk_docs}")
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader