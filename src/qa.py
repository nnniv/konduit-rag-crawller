from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

LOG = logging.getLogger("qa")


def chroma_retrieve(query: str, embedding_model: str, top_k: int = 5, collection: str = "default", persist_dir: str = "data/chroma") -> List[Tuple[Dict[str, Any], float]]:
    embeddings = OllamaEmbeddings(model=embedding_model, base_url="http://127.0.0.1:11434")
    vs = Chroma(collection_name=collection, embedding_function=embeddings, persist_directory=persist_dir)
    docs_scores = vs.similarity_search_with_score(query, k=top_k)
    results: List[Tuple[Dict[str, Any], float]] = []
    for doc, score in docs_scores:
        meta = doc.metadata or {}
        meta = {**meta, "text": doc.page_content}
        results.append((meta, score))
    return results


def ask_question(question: str, top_k: int = 5, embedding_model: str = "embeddinggemma", generation_model: str = "gemma3:latest") -> Dict[str, Any]:
    t0 = time.monotonic()
    pairs = chroma_retrieve(question, embedding_model, top_k=top_k)
    t1 = time.monotonic()
    if not pairs:
        timings = {
            "retrieval_ms": int((t1 - t0) * 1000),
            "generation_ms": 0,
            "total_ms": int((t1 - t0) * 1000),
        }
        return {
            "answer": "I couldn't find relevant information in the index.",
            "sources": [],
            "timings": timings,
        }

    # Build context
    sources: List[Dict[str, str]] = []
    context_sections: List[str] = []
    for i, (meta, _score) in enumerate(pairs, start=1):
        url = meta.get("url", "")
        text = (meta.get("text", "") or "")
        snippet = text[:800]
        context_sections.append(f"[{i}] URL: {url}\n{snippet}")
        if url:
            sources.append({"url": url, "snippet": snippet})

    system_prompt = (
        "You are a helpful assistant that answers strictly using the provided CONTEXT. "
        "If the answer is not present in the context, say you don't know. "
        "Cite sources inline as [n] where n corresponds to the numbered context section. Keep the answer concise."
    )
    context_block = "\n\n".join(context_sections)
    user_prompt = (
        f"QUESTION: {question}\n\n"
        f"CONTEXT (numbered sections):\n{context_block}\n\n"
        "Write the best possible answer using only the context above. Include inline citations [n] after the statements you derive."
    )

    llm = ChatOllama(model=generation_model, base_url="http://127.0.0.1:11434", temperature=0.1)
    t2 = time.monotonic()
    msg = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    t3 = time.monotonic()

    answer_text = getattr(msg, "content", "") or ""
    timings = {
        "retrieval_ms": int((t1 - t0) * 1000),
        "generation_ms": int((t3 - t2) * 1000),
        "total_ms": int((t3 - t0) * 1000),
    }
    return {"answer": answer_text, "sources": sources, "timings": timings}
