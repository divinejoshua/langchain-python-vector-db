import os
import sys

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_openai import OpenAI
from langchain_community.vectorstores import Chroma
from decouple import config
import chromadb
from chromadb.config import Settings


os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings()

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

# Connect to chromadb client
client = chromadb.HttpClient(settings=Settings(allow_reset=True))
print(client.get_collection("my_collection_two"))


db = Chroma(
    client=client,
    collection_name="my_collection_two",
    embedding_function=embeddings,
)


chain = RetrievalQA.from_llm(
  llm=OpenAI(model="gpt-4o"),
  retriever=db.as_retriever(),
)


#Query
query = "How old is rema?"
chat_history = []
result = chain.invoke({'query': query})
print(result['result'])
