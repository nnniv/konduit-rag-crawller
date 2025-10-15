# RAG Crawler

This project implements a Retrieval-Augmented Generation (RAG) system that, starting from a given website URL, crawls all accessible pages within the domain, indexes their textual content, and provides an API to answer questions based strictly on the crawled data with citations to original source URLs.

## Features

- Polite, domain-limited web crawler respecting `robots.txt` to fetch HTML content (up to a configurable page limit).
- Content extraction with boilerplate removal and text cleaning.
- Text chunking and embedding with open-source embedding models for vector similarity search.
- Vector index storage using Chroma for efficient retrieval.
- FastAPI HTTP API exposing endpoints to crawl, index, and ask questions with citations and timing info.
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
│   └── qa.py              # Question-Answering logic
└── uv.lock
```
## Getting Started

### Prerequisites

- Docker and Docker Compose installed.

### Build and Run

Build the Docker image and start the services:

```bash
docker-compose up --build
```

This will initialize the FastAPI server, accessible on port `3400`.

### API Usage

- `/crawl` (POST): Initiate crawling from a start URL.
- `/index` (POST): Create or update the vector index with crawled content.
- `/ask` (POST): Submit questions to retrieve answers grounded in crawled data, with citations to source URLs.

Sample requests and responses are documented in the [`examples/EXAMPLES.md`](https://github.com/nnniv/konduit-rag-crawller/blob/main/examples/EXAMPLES.md).

## Models Used

This project employs the following key models and frameworks for the RAG pipeline:

- **Embedding Model:** Uses the `embeddingsgemma` model via the `OllamaEmbeddings` interface from the `langchaincommunity` package. This model generates dense vector embeddings for textual chunks extracted during crawling, which are critical for semantic search during question answering.

- **Question Answering Model:** Utilizes the `gemma3latest` chat model through `ChatOllama`. This LLM generates answers based on retrieved context chunks, enforcing strict grounding by only responding with information supported by the indexed content, or declining otherwise.

- **Vector Store:** The Chroma open-source vector database stores embeddings for efficient similarity search.

- **API Framework:** The system exposes its functionality via a **FastAPI** server, providing RESTful endpoints for crawling, indexing, and querying. FastAPI supports asynchronous processing and automatic API documentation, contributing to faster development and responsive interfaces.

## Pros and Cons of This Implementation

### Pros

- Open-source, self-hosted models (Ollama gemma family) avoid reliance on expensive external LLM services, reducing operational costs.
- Strict grounding in answers minimizes hallucinations and improves reliability.
- Modular design enables easy swaps or upgrades of embedding and generation models.
- FastAPI offers modern, asynchronous API support with automatic docs and high performance.
- Chroma vector store allows quick semantic retrieval with a lightweight local deployment.
- Polite crawler respects robots.txt and supports crawl constraints for ethical crawling.

### Cons

- Limited to single-domain crawling with a set page limit, limiting scale for very large or multi-site crawls.
- Embedded models may have lower accuracy or larger latency compared to proprietary cloud LLMs.
- QA relies on retrieved content quality; if crawl misses key info, answers may be incomplete.
- Local hosting of models requires sufficient compute resources (CPU/GPU) which may not scale for heavy usage.
- Lack of user interface; interaction limited to API or CLI clients.

## System Specifications

The system is designed to operate efficiently on a typical modern development machine with the following specs:

- **CPU**: 11th Gen Intel i5-1135G7 (8 cores) @ 4.2 GHz
- **GPU**: Intel TigerLake-LP GT2 [Iris Xe Graphics]
- **Memory**: 16 GB RAM

This setup was used for development and testing, providing a good balance for local deployment and experimentation with models and data.


