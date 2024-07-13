# import
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import os
from decouple import config


# load the document and split it into chunks
loader = TextLoader("data/nigeria.txt")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function)


# Check if data exist
if os.path.exists("chroma_db"):
    # load from disk
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
else:
    # save to disk
    db = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")

# query it
query = "What happened in 1962"
docs = db.similarity_search(query)

# print results
print(docs[0])