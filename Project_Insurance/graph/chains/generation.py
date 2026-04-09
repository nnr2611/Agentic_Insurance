from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from constants import SYSTEM_PROMPT_GENERATION

llm = ChatOpenAI(temperature=0)
prompt = PromptTemplate.from_template(SYSTEM_PROMPT_GENERATION)
generation_chain = prompt | llm | StrOutputParser()