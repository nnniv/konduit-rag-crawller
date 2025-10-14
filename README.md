Here is the updated README with the detailed models section included:

---

# RAG Crawl-Index-QA Assessment

This project implements a Retrieval-Augmented Generation (RAG) system that, starting from a given website URL, crawls all accessible pages within the domain, indexes their textual content, and provides an API to answer questions based strictly on the crawled data with citations to original source URLs. The goal is to demonstrate practical skills in web crawling, text indexing, retrieval, grounded QA, and engineering clarity.

## Feature

- Polite, domain-limited web crawler respecting `robots.txt` to fetch HTML content (up to a configurable page limit).
- Content extraction with boilerplate removal and text cleaning.
- Text chunking and embedding with open-source embedding models for vector similarity search.
- Vector index storage using Chroma for efficient retrieval.
- FastAPI HTTP API exposig endpoints to crawl, index, and ask questions with citations and timing info.
- Robust grounding: answers are provided only if supported by retrieved context with explicit refusals otherwise.
- Logging and basic observability of retrieval and generation latencies.

## Project Structure

```
.
├── data
│   ├── chroma             # Persisted vector index data
│   ├── crawl              # Raw crawl result files (JSONL)
│   ├── crawls             # History or other crawl data
│   └── index              # Processed index files
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── src
│   ├── api
│   │   └── app.py         # API
│   ├── crawler.py         # Polite web crawler implementation
│   ├── indexer.py         # Text chunking, embedding, and vector index management
│   └── qa.py              # Question-Answering logic using retrieval from vector index
└── uv.lock
```

## Models Used

This project leverages open-source embedding and language models to implement the core RAG pipeline, ensuring cost-effective and transparent operation:

### Embedding Model

- **OllamaEmbeddings**: The embedding model used to convert textual chunks into dense vector representations is provided by OllamaEmbeddings from the `langchaincommunity` library.
- The embedding model converts cleaned and chunked crawl content into vectors that are stored and indexed in the Chroma vector store.
- This vector representation enables high-quality semantic similarity search during question answering.

### Question Answering / Generation Model

- **ChatOllama (gemma3latest)**: For generating answers, the project uses an Ollama-based chat model, specifically the "gemma3latest" model, called via the `ChatOllama` wrapper.
- The question answering module retrieves the most relevant chunks (top-K) from the vector index based on semantic similarity of embeddings.
- It builds a prompt containing retrieved context passages and enforces grounded responses by instructing the model to only answer from the explicit context or refuse with "I don't know" if insufficient information is found.
- The model operates asynchronously over HTTP with a configurable base URL for the Ollama API server, enabling flexible deployment scenarios.

### Vector Store

- **Chroma**: A locally persisted, open-source vector database is used for storing the embeddings. It provides efficient similarity search functionality integral to retrieval in RAG.

### Design Rationale

- Using Ollama models supports open ecosystem usage without depending on costly cloud APIs.
- Embedding and generation models can be independently swapped or upgraded.
- Enforcement of strict grounding in the prompt is critical to maintain the reliability of answers.
- Coupling retrieval with generation models follows modern best practices in building RAG pipelines.

## System Specifications

The system is designed to operate efficiently on a typical modern development machine with the following specs:

- **CPU**: 11th Gen Intel i5-1135G7 (8 cores) @ 4.2 GHz
- **GPU**: Intel TigerLake-LP GT2 [Iris Xe Graphics]
- **Memory**: 16 GB RAM

This setup was used for development and testing, providing a good balance for local deployment and experimentation with models and data.

## Getting Started

### Prerequisites

- Docker and Docker Compose installed.
- Follow the build and run instructions below.

### Build and Run

Build the Docker image and start the services:

```bash
docker-compose up --build
```

This will initialize the FastAPI server, accessible on port 8000.

### API Usage

- `/crawl` (POST): Initiate crawling from a start URL.
- `/index` (POST): Create or update the vector index with crawled content.
- `/ask` (POST): Submit questions to retrieve answers grounded in crawled data, with citations to source URLs.

Sample requests and responses are documented in the examples/EXAMPLES.md.
