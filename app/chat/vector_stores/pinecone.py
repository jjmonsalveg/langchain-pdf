import os
import pinecone
from langchain.vectorstores.pinecone import Pinecone
from app.chat.embeddings.openai import embeddings

pinecone.Pinecone(
    api_key=os.getenv("PINE_CONE_API_KEY"), environment=os.getenv("PINE_CONE_ENV_NAME")
)

# We had issues with PINECONE_API_KEY for some reason always were blank
# Pinecone.from_existing_index will look into PINECONE_API_KEY so we need to
# set it manually
os.environ["PINECONE_API_KEY"] = os.getenv("PINE_CONE_API_KEY")
vector_stores = Pinecone.from_existing_index(
    os.getenv("PINE_CONE_INDEX_NAME"), embeddings
)