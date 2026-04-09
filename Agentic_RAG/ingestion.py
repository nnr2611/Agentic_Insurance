from dotenv import load_dotenv
load_dotenv()

#THis is used to split up our documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
#We will be using a web based loading to load the documents from the internet
from langchain_community.document_loaders import WebBaseLoader

#Vector store
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

#plugging in each url - we will ahve the list of langchain documents
#what you get back is a list with a single element which in turn is a list representing one doc per url
docs = [WebBaseLoader(url).load() for url in urls]

#We want to flatten that list - so we will be itearting through the docs - and each item here is a sublist and each item in that sublist is the doc we want
docs_list = [item for sublist in docs for item in sublist]

#We will now split it up into chunks, we want our chunk size to be 250 with no overlap
#Basic text splitting definitions
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

#We will use the text splitter to split the documents in document list
#SO eventually - we will get from it a list of smaller shunks - that represent the d_ocuments
doc_splits = text_splitter.split_documents(docs_list)

#We will index them into chromaDB - which will run locally in our machine- Autoencoding
#Chroma object - its going to have from_dcoumetns method - will take the documents - in the form of chunks
#We are going to call collection name in our index as - rag-chroma
#We want to use OpenAI embeddings -

#WE WILL NOW PERSIST THE DATASTORE INTO OUR DISC - it will persist at that location - after it does indexing
#ONLY the first time we will run this - after that we dont want to run indexing everytime we run this file, we will just simply want to load the indexing from the disks - form VectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
    persist_directory="./.chroma",
)

#Langchain retriever object - to indexed data from the chromaDB
#It will intialise an obejct of Chroma class and used that as retriever method in order to turn it into a langchain retriever
#So we will be able to perform similarity searches using that 
retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(),
).as_retriever()