from typing import Any, Dict
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from graph.state import GraphState
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("-- LLM Web Search Simulation --")
    question = state['question']
    documents = state['documents']

    response = llm.invoke(
        f"Provide a concise and reliable background overview about the topic: '{question}'. Focus on technical and factual information."
    )

    # Wrap the LLM's answer inside a LangChain Document
    web_results = Document(
        page_content=response.content,
    )

    # Since this documents - we are getting after filtering out irrelevant docs - this list might or might not contain any docs so based on that we are appending or initializing the web results to the docs
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents":documents,"question":question}

if __name__ == "__main__":
    output = web_search(state={"question": "agent memory", "documents": None})
    print(output["documents"][0].page_content)
