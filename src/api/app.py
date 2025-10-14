from __future__ import annotations

import asyncio
import logging
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, AnyHttpUrl, Field

from ..crawler import PoliteCrawler

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
        return CrawlResponse(page_count=len(results), skipped_count=crawler.skipped_count, urls=urls)

    try:
        resp = await asyncio.to_thread(run_crawl)
        return resp
    except Exception as e:
        LOG.exception("crawl failed")
        raise HTTPException(status_code=500, detail=str(e))
