from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

#For structured output
class GraderHallucinations(BaseModel):
    "Binary score for hallucination present in the generation answer."
    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )
    #Because we set binary score type to be boolean - langhcain otuput parser - it will parse the LLm ansewr into just a Boolean

#The answer we get back from LLM - the langcain will formati t as pydantic class of gradehallucinations - will have just one attribute
structured_llm_grader = llm.with_structured_output(GraderHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of documents.
        Give a binary score of 'yes' or 'no' where Yes means that the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human","Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
    ]
)

#This is the definition of hallucination grader chain that takes in the hallucination prompt and pipes it with the structed llm grader
hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader