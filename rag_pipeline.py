import os
from dotenv import load_dotenv

from cohere import ClientV2
from pinecone import Pinecone

# Load environment variables
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize clients
co = ClientV2(api_key=COHERE_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


def embed_query(question):
    """
    Convert a user question into an embedding.
    """

    response = co.embed(
        model="embed-english-v3.0",
        input_type="search_query",
        texts=[question],
        embedding_types=["float"]
    )

    return response.embeddings.float[0]


def retrieve_context(question, top_k=3, namespace=None):
    """
    Retrieve the most relevant chunks from Pinecone.
    """

    query_embedding = embed_query(question)

    results = index.query(
    vector=query_embedding,
    top_k=top_k,
    include_metadata=True,
    namespace=namespace
    )

    return results

def generate_answer(question, results):
    """
    Generate an answer using the retrieved context.
    """

    context = ""

    for match in results.matches:
        context += match.metadata["text"] + "\n\n"

    response = co.chat(
        model="command-r-08-2024",
        messages=[
            {
                "role": "user",
                "content": f"""
                Answer the question ONLY using the context below.

                If the answer is not available in the context, reply:
                "I couldn't find that information in the provided document."

                Context:
                {context}
                Question:
                {question}
                """
            }
        ]
    )

    return response.message.content[0].text


if __name__ == "__main__":

    question = input("Enter your question: ")

    results = retrieve_context(question)
    print("\n========== RETRIEVAL VALIDATION ==========\n")

    for i, match in enumerate(results.matches, start=1):
        print(f"Result {i}")
        print(f"Similarity Score : {match.score:.4f}")
        print(f"Source           : {match.metadata['source']}")
        print("-" * 50)

    answer = generate_answer(question, results)

    print("\nAnswer:\n")
    print(answer)