from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document  # <<< NEW!!

def ingest_text_to_vectordb(original_text: str) -> None:
    """
    Ingests given text into the Chroma VectorDB after splitting into Document objects.
    """

    # 1. Split text
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250,
        chunk_overlap=0
    )
    doc_splits = text_splitter.split_text(original_text)

    # 2. Wrap each split into a Document
    documents = [Document(page_content=chunk) for chunk in doc_splits]

    # 3. Store into Chroma
    vectorstore = Chroma.from_documents(
        documents=documents,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(),
        persist_directory="./.chroma",
    )

# retriever at module level
retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(),
).as_retriever()
