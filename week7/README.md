# Document Question Answering System (RAG)

## Live Demo

Streamlit App:

https://document-app-rag-7zrsnhmephfzjq7zbvywwv.streamlit.app/

## Overview

This project implements a Retrieval-Augmented Generation (RAG) based Document Question Answering System.

The system allows users to upload custom PDF documents, extracts relevant information, converts the document content into vector embeddings, stores them in a vector database, and generates context-aware answers using a language model.

Unlike traditional question-answering systems, this approach retrieves relevant information from the uploaded document before generating the response, improving accuracy and grounding.

## Objectives

* Build a document ingestion pipeline for custom documents.
* Process unstructured text using chunking techniques.
* Generate embeddings using a pre-trained embedding model.
* Store embeddings in a vector database.
* Retrieve relevant document chunks for user queries.
* Generate grounded answers using retrieved context.

## System Architecture

```
User Uploads Document
        |
        ↓
PDF Text Extraction
        |
        ↓
Text Chunking
        |
        ↓
Cohere Embedding Generation
        |
        ↓
Pinecone Vector Database
        |
        ↓
User Query
        |
        ↓
Similarity Search
        |
        ↓
Retrieved Context
        |
        ↓
Cohere Language Model
        |
        ↓
Final Answer
```

## Technologies Used

* Python
* Streamlit
* PyPDF2
* LangChain Text Splitters
* Cohere Embeddings
* Pinecone Vector Database
* Cohere Command-R Language Model

## Features

* Upload custom PDF documents.
* Automatically extract and process document text.
* Split documents into smaller chunks.
* Generate semantic embeddings.
* Store vectors in Pinecone.
* Retrieve relevant document sections.
* Generate context-aware answers.
* Display retrieved context and similarity scores.
* Display system metrics.

## Project Structure

```
Document-QA-RAG/

├── app.py
├── ingest.py
├── rag_pipeline.py
├── requirements.txt
├── system_metrics.txt
├── README.md
│
└── utils/
    ├── pdf_loader.py
    └── ingestion.py
```

## How to Run Locally

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```
COHERE_API_KEY=your_cohere_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=document-qa-rag
```

### Run Application

```bash
streamlit run app.py
```

## Example Query

Question:

```
What programming languages are mentioned in the document?
```

The system retrieves relevant document chunks and generates an answer based on the retrieved information.

## System Metrics

* Embedding Model: Cohere embed-english-v3.0
* Embedding Dimension: 1024
* Vector Database: Pinecone
* Similarity Metric: Cosine Similarity
* Language Model: Cohere Command-R
* Chunking Method: Recursive Character Text Splitter
* Chunk Size: 500
* Chunk Overlap: 100

## Conclusion

This project demonstrates the complete workflow of a Retrieval-Augmented Generation system, including document ingestion, text processing, embedding generation, vector storage, retrieval, and grounded response generation.

The system can be used for document assistants, knowledge management systems, and domain-specific question-answering applications.

