import os
from uuid import uuid4
import streamlit as st

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cohere import ClientV2
from pinecone import Pinecone


load_dotenv()


COHERE_API_KEY = os.getenv("COHERE_API_KEY") or st.secrets["COHERE_API_KEY"]
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or st.secrets["PINECONE_INDEX_NAME"]


co = ClientV2(api_key=COHERE_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(PINECONE_INDEX_NAME)



def extract_text_from_pdf(file):
    """
    Extract text from uploaded PDF.
    """

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text



def chunk_text(text, chunk_size=500, chunk_overlap=100):
    """
    Split text into smaller chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_text(text)

    return chunks



def create_embeddings(chunks):
    """
    Generate Cohere embeddings.
    """

    response = co.embed(
        model="embed-english-v3.0",
        input_type="search_document",
        texts=chunks,
        embedding_types=["float"]
    )

    return response.embeddings.float



def store_embeddings(chunks, embeddings, namespace):
    """
    Store vectors in Pinecone namespace.
    """

    vectors = []

    for chunk, embedding in zip(chunks, embeddings):

        vectors.append(
            {
                "id": str(uuid4()),
                "values": embedding,
                "metadata": {
                    "text": chunk
                }
            }
        )


    index.upsert(
        vectors=vectors,
        namespace=namespace
    )


    return len(vectors)