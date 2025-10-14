from __future__ import annotations

import asyncio
import logging
from typing import List

import time
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, AnyHttpUrl, Field

from ..crawler import PoliteCrawler
from ..indexer import index_into_chroma_latest, ensure_dirs
from ..qa import ask_question

LOG = logging.getLogger("api")
app = FastAPI(title="Crawler API")


class CrawlRequest(BaseModel):
    start_url: AnyHttpUrl
    max_pages: int = Field(40, ge=1, le=200)
    max_depth: int = Field(3, ge=0, le=10)
    crawl_delay_ms: int = Field(1000, ge=0, le=60000)


class CrawlResponse(BaseModel):
    page_count: int
    skipped_count: int
    urls: List[AnyHttpUrl]


@app.post("/crawl", response_model=CrawlResponse)
async def crawl_endpoint(req: CrawlRequest):
    delay = req.crawl_delay_ms / 1000.0

    def run_crawl() -> CrawlResponse:
        crawler = PoliteCrawler(str(req.start_url), max_pages=req.max_pages, max_depth=req.max_depth, delay=delay)
        results = crawler.crawl()
        urls = [r.url for r in results]
        # persist crawl results so indexing step can use latest file
        ensure_dirs()
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_path = Path("data/crawl") / f"crawl-{ts}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for r in results:
                json.dump({
                    "url": r.url,
                    "title": r.title,
                    "text": r.text,
                    "fetched_at": r.fetched_at,
                }, fh, ensure_ascii=False)
                fh.write("\n")
        return CrawlResponse(page_count=len(results), skipped_count=crawler.skipped_count, urls=urls)

    try:
        resp = await asyncio.to_thread(run_crawl)
        return resp
    except Exception as e:
        LOG.exception("crawl failed")
        raise HTTPException(status_code=500, detail=str(e))


class IndexRequest(BaseModel):
    chunk_size: int = Field(800, ge=100, le=5000)
    chunk_overlap: int = Field(100, ge=0, le=4000)
    embedding_model: str = Field("gemma3:embed", description="Ollama embedding model, e.g., 'gemma3:embed' (EmbeddingGemma), 'nomic-embed-text', etc.")


class IndexResponse(BaseModel):
    vector_count: int
    errors: list[str]


@app.post("/index", response_model=IndexResponse)
async def index_endpoint(req: IndexRequest):
    # Basic sanity: overlap < size
    if req.chunk_overlap >= req.chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap must be less than chunk_size")

    def run_index() -> IndexResponse:
        count, errs = index_into_chroma_latest(req.chunk_size, req.chunk_overlap, req.embedding_model)
        return IndexResponse(vector_count=count, errors=errs)

    try:
        resp = await asyncio.to_thread(run_index)
        return resp
    except Exception as e:
        LOG.exception("index failed")
        raise HTTPException(status_code=500, detail=str(e))


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(5, ge=1, le=20)


class AskSource(BaseModel):
    url: AnyHttpUrl
    snippet: str


class AskTimings(BaseModel):
    retrieval_ms: int
    generation_ms: int
    total_ms: int


class AskResponse(BaseModel):
    answer: str
    sources: list[AskSource]
    timings: AskTimings


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(req: AskRequest):
    def run_ask() -> AskResponse:
        payload = ask_question(req.question, top_k=req.top_k)
        sources = [AskSource(url=s["url"], snippet=s["snippet"]) for s in payload.get("sources", []) if s.get("url")]
        t = payload.get("timings", {})
        timings = AskTimings(
            retrieval_ms=int(t.get("retrieval_ms", 0)),
            generation_ms=int(t.get("generation_ms", 0)),
            total_ms=int(t.get("total_ms", 0)),
        )
        return AskResponse(answer=str(payload.get("answer", "")), sources=sources, timings=timings)

    try:
        resp = await asyncio.to_thread(run_ask)
        return resp
    except Exception as e:
        LOG.exception("ask failed")
        raise HTTPException(status_code=500, detail=str(e))
