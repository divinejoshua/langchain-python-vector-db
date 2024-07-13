import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import uuid
from chromadb.config import Settings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from decouple import config
import os

os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')

# load the document and split it into chunks
loader = TextLoader("data/nigeria.txt")
documents = loader.load()

embeddings = OpenAIEmbeddings()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


student_info = """
Alexandra Thompson, a 19-year-old computer science sophomore with a 3.7 GPA,
is a member of the programming and chess clubs who enjoys pizza, swimming, and hiking
in her free time in hopes of working at a tech company after graduating from the University of Washington.
"""

club_info = """
The university chess club provides an outlet for students to come together and enjoy playing
the classic strategy game of chess. Members of all skill levels are welcome, from beginners learning
the rules to experienced tournament players. The club typically meets a few times per week to play casual games,
participate in tournaments, analyze famous chess matches, and improve members' skills.
"""

university_info = """
The University of Washington, founded in 1861 in Seattle, is a public research university
with over 45,000 students across three campuses in Seattle, Tacoma, and Bothell.
As the flagship institution of the six public universities in Washington state,
UW encompasses over 500 buildings and 20 million square feet of space,
including one of the largest library systems in the world.
Rema is 6 years old
"""

# Connect to chromadb client
client = chromadb.HttpClient(settings=Settings(allow_reset=True))
client.reset()  # resets the database
collection = client.create_collection("my_collection_two")
collection.add(
    documents = [student_info, club_info, university_info],
    metadatas = [{"source": "student info"},{"source": "club info"},{'source':'university info'}],
    ids = ["id1", "id2", "id3"]
)
# for doc in docs:
#     collection.add(
#         ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
#     )


query = "What happened in 1962"
# docs = db.similarity_search(query)
print(docs[0].page_content)