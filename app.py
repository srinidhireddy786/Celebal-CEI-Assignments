import streamlit as st

from utils.ingestion import (
    extract_text_from_pdf,
    chunk_text,
    create_embeddings,
    store_embeddings
)

from rag_pipeline import retrieve_context, generate_answer


st.set_page_config(
    page_title="Document Question Answering System",
    page_icon="📄"
)


st.title("📄 Document Question Answering System (RAG)")


# Store namespace
if "namespace" not in st.session_state:
    st.session_state.namespace = None


st.subheader("Upload Document")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


if uploaded_file:

    if st.button("Process Document"):

        with st.spinner("Processing document..."):

            # Extract text
            text = extract_text_from_pdf(uploaded_file)


            # Create chunks
            chunks = chunk_text(text)


            # Create embeddings
            embeddings = create_embeddings(chunks)


            # Namespace using file name
            namespace = uploaded_file.name.replace(".pdf", "")

            # Store vectors
            count = store_embeddings(
                chunks,
                embeddings,
                namespace
            )


            st.session_state.namespace = namespace


            st.success(
                f"Document processed successfully! {count} chunks stored."
            )



st.divider()


st.subheader("Ask a Question")


question = st.text_input(
    "Enter your question"
)


if st.button("Get Answer"):

    if st.session_state.namespace is None:

        st.warning(
            "Please upload and process a document first."
        )

    elif question:

        with st.spinner("Generating answer..."):

            results = retrieve_context(
                question,
                namespace=st.session_state.namespace
            )
            answer = generate_answer(
                question,
                results
            )
            st.subheader("Answer")
            st.write(answer)

            st.subheader("Retrieval Validation")

            for i, match in enumerate(results.matches, start=1):

                st.write(f"Result {i}")
                st.write(f"Similarity Score: {match.score:.4f}")
                st.write(match.metadata["text"])
                st.write("---")



            with st.expander("Retrieved Context"):

                for match in results.matches:

                    st.write(
                        match.metadata["text"]
                    )
st.divider()

st.subheader("System Metrics")

metrics = """
Embedding Model: Cohere embed-english-v3.0

Embedding Dimension: 1024

Vector Database: Pinecone

Similarity Metric: Cosine

Language Model: Cohere command-r-08-2024

Chunking Method: Recursive Character Text Splitter

Chunk Size: 500

Chunk Overlap: 100

Document Type: PDF
"""

st.text(metrics)