from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os

LOG = logging.getLogger("indexer")


@dataclass
class Chunk:
    url: str
    chunk_index: int
    text: str

try:
    from langchain_ollama import OllamaEmbeddings  # new modular package
except ImportError:  # fallback
    from langchain_community.embeddings import OllamaEmbeddings  # type: ignore

try:
    from langchain_chroma import Chroma  # new modular package
except ImportError:  # fallback
    from langchain_community.vectorstores import Chroma  # type: ignore


def ensure_dirs():
    Path("data/crawl").mkdir(parents=True, exist_ok=True)
    Path("data/index").mkdir(parents=True, exist_ok=True)
    Path("data/chroma").mkdir(parents=True, exist_ok=True)


def latest_crawl_file() -> Path | None:
    ensure_dirs()
    files = list(Path("data/crawl").glob("*.jsonl"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def read_crawl_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def clean_text(text: str) -> str:
    # normalize whitespace, drop very long runs of newlines
    t = re.sub(r"\s+", " ", text).strip()
    return t


def build_chunks_from_crawl(path: Path, size: int, overlap: int) -> Tuple[List[str], List[Dict[str, Any]]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    for rec in read_crawl_jsonl(path):
        url = rec.get("url", "")
        raw_text = rec.get("text", "")
        body = clean_text(raw_text)
        if not body:
            continue
        chunks = splitter.split_text(body)
        for idx, ch in enumerate(chunks):
            texts.append(ch)
            metadatas.append({
                "url": url,
                "chunk_index": idx,
            })
    return texts, metadatas


def index_into_chroma_latest(chunk_size: int, chunk_overlap: int, embedding_model: str, collection: str = "default", persist_dir: str = "data/chroma") -> Tuple[int, List[str]]:
    ensure_dirs()
    crawl_path = latest_crawl_file()
    if not crawl_path:
        return 0, ["No crawl data found in data/crawl"]

    texts, metadatas = build_chunks_from_crawl(crawl_path, chunk_size, chunk_overlap)
    if not texts:
        return 0, ["No text chunks produced from latest crawl"]

    try:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
    except Exception as e:
        return 0, [f"Failed to initialize embeddings for model '{embedding_model}': {e}"]

    try:
        vs = Chroma(collection_name=collection, embedding_function=embeddings, persist_directory=persist_dir)
        batch = 64
        total = 0
        for i in range(0, len(texts), batch):
            j = min(len(texts), i + batch)
            vs.add_texts(texts=texts[i:j], metadatas=metadatas[i:j])
            total += (j - i)
        # vs.persist()  # Deprecated call removed
        return total, []
    except Exception as e:
        LOG.exception("Chroma indexing failed")
        return 0, [str(e)]