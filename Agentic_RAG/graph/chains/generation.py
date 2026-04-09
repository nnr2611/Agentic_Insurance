from langchain import hub
#This StrOutputParser - is simply going to get the output of a LLM and convert that into a string
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

#creating an llm instance
llm = ChatOpenAI(temperature=0)

#This is a very standard rag-prompt from the langchain team
#If you look at this prompt template from the huggingface it will say - 
#You are an assistant for question-asnwering tasks, Use the following peices of retrieved  context to answer the question. If you dont know the answer, just say that you dont know. Use three sentences max and keep the answers concise.
prompt = hub.pull("rlm/rag-prompt")

#We are piping the prompt into the llm and piping the LLM results into the StrOutputParser format
#SO IF WE INVOKE THIS GENERATION CHAIN WITH THE DOCS WE RETRIEVED FROM RAG AND THE QUESTION FROM THE USER - WE SHOULD GET A PROPER ANSWER REFERRING FROM THE DOCS
generation_chain = prompt | llm | StrOutputParser()