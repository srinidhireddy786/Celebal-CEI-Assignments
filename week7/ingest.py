import os
from uuid import uuid4

from dotenv import load_dotenv
from cohere import ClientV2
from pinecone import Pinecone

from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.pdf_loader import load_documents

# Load environment variables
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize clients
co = ClientV2(api_key=COHERE_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


def chunk_documents(chunk_size=500, chunk_overlap=100):
    """
    Load documents and split them into chunks.
    """
    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []

    for doc in documents:
        split_text = splitter.split_text(doc["text"])

        for text in split_text:
            chunks.append({
                "text": text,
                "source": doc["filename"]
            })

    return chunks


def create_embeddings(chunks):
    """
    Generate Cohere embeddings for all chunks.
    """

    texts = [chunk["text"] for chunk in chunks]

    response = co.embed(
        model="embed-english-v3.0",
        input_type="search_document",
        texts=texts,
        embedding_types=["float"]
    )

    embeddings = response.embeddings.float

    return embeddings


def store_vectors(chunks, embeddings):
    """
    Store embeddings in Pinecone.
    """

    vectors = []

    for chunk, embedding in zip(chunks, embeddings):

        vectors.append(
            {
                "id": str(uuid4()),
                "values": embedding,
                "metadata": {
                    "text": chunk["text"],
                    "source": chunk["source"]
                }
            }
        )

    index.upsert(vectors=vectors)

    print(f"\nSuccessfully stored {len(vectors)} vectors in Pinecone.")


if __name__ == "__main__":

    print("\nLoading and Chunking Documents...")

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100

    chunks = chunk_documents(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    print("\n========== VALIDATION LOGS ==========")
    print(f"Documents Loaded : {len(load_documents())}")
    print(f"Total Chunks     : {len(chunks)}")
    print("\nChunking Configuration")
    print(f"Chunk Size    : {CHUNK_SIZE}")
    print(f"Chunk Overlap : {CHUNK_OVERLAP}")

    print("\nGenerating Embeddings...")

    embeddings = create_embeddings(chunks)
    print(f"Embeddings Generated : {len(embeddings)}")
    print(f"Embedding Dimension  : {len(embeddings[0])}")

    print(f"Embeddings Created: {len(embeddings)}")

    print("\nUploading to Pinecone...")

    store_vectors(chunks, embeddings)
    print("Vector Store : Pinecone")
    print("Embedding Model : embed-english-v3.0")
    print("\nDocument ingestion completed successfully.")
