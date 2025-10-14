from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Set

import requests
import tldextract
from bs4 import BeautifulSoup, Comment
from readability import Document
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser


LOG = logging.getLogger("crawler")


@dataclass
class PageResult:
    url: str
    title: Optional[str]
    text: str
    fetched_at: str


class PoliteCrawler:
    def __init__(self, start_url: str, max_pages: int = 40, max_depth: int = 3, delay: float = 1.0, user_agent: str = "konduit-rag-crawler/0.1"):
        self.start_url = start_url
        self.max_pages = max_pages
        # maximum link depth from the start_url (start is depth 0)
        self.max_depth = max_depth
        self.delay = delay
        self.user_agent = user_agent

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

        self.visited: Set[str] = set()
        self.to_visit = queue.Queue()
        self.to_visit.put((start_url, 0))

        self.skipped_count = 0

        self.start_reg_domain = self._registrable_domain(start_url)
        self.robots = self._fetch_robots(start_url)

    def _registrable_domain(self, url: str) -> str:
        parts = tldextract.extract(url)
        return parts.registered_domain or parts.domain or url

    def _fetch_robots(self, url: str) -> Optional[RobotFileParser]:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = RobotFileParser()
        try:
            rp.set_url(robots_url)
            rp.read()
            LOG.debug("Loaded robots.txt from %s", robots_url)
            return rp
        except Exception:
            LOG.debug("Could not load robots.txt from %s", robots_url)
            return None

    def _allowed_by_robots(self, url: str) -> bool:
        if not self.robots:
            return True
        try:
            return self.robots.can_fetch(self.user_agent, url)
        except Exception:
            return True

    def _same_registrable_domain(self, url: str) -> bool:
        return self._registrable_domain(url) == self.start_reg_domain

    def _clean_html(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript", "header", "footer", "aside", "svg"]):
            tag.decompose()
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        return str(soup)

    def _extract_main(self, html: str, url: str) -> PageResult:
        try:
            doc = Document(html)
            title = doc.short_title()
            content_html = doc.summary()
            cleaned = self._clean_html(content_html)
            text = BeautifulSoup(cleaned, "lxml").get_text(separator="\n", strip=True)
            if len(text) < 200:
                raise ValueError("extracted too short")
            return PageResult(url=url, title=title, text=text, fetched_at=datetime.utcnow().isoformat())
        except Exception:
            soup = BeautifulSoup(self._clean_html(html), "lxml")
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else None
            candidate = soup.find("main") or soup.find("article") or soup.body
            text = candidate.get_text(separator="\n", strip=True) if candidate else soup.get_text(separator="\n", strip=True)
            return PageResult(url=url, title=title, text=text or "", fetched_at=datetime.utcnow().isoformat())

    def crawl(self):
        results = []
        last_fetch_time = 0.0

        while not self.to_visit.empty() and len(results) < self.max_pages:
            url, depth = self.to_visit.get()
            if url in self.visited:
                self.skipped_count += 1
                continue
            if not self._same_registrable_domain(url):
                LOG.debug("Skipping out-of-domain %s", url)
                self.skipped_count += 1
                continue
            if not self._allowed_by_robots(url):
                LOG.debug("Disallowed by robots: %s", url)
                self.skipped_count += 1
                continue

            elapsed = time.time() - last_fetch_time
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)

            LOG.info("Fetching %s", url)
            try:
                resp = self.session.get(url, timeout=15)
                last_fetch_time = time.time()
            except Exception as e:
                LOG.debug("Failed to fetch %s: %s", url, e)
                self.visited.add(url)
                self.skipped_count += 1
                continue

            if resp.status_code != 200 or 'text/html' not in resp.headers.get('Content-Type', ''):
                LOG.debug("Skipping non-html or bad status for %s: %s %s", url, resp.status_code, resp.headers.get('Content-Type'))
                self.visited.add(url)
                self.skipped_count += 1
                continue

            page = self._extract_main(resp.text, url)
            results.append(page)
            self.visited.add(url)

            # Find and queue links
            soup = BeautifulSoup(resp.text, "lxml")
            # queue links with depth control
            if depth < self.max_depth:
                for a in soup.find_all("a", href=True):
                    href = a.get("href")
                    if not isinstance(href, str):
                        continue
                    # normalize
                    joined = urljoin(url, href)
                    parsed = urlparse(joined)
                    if parsed.scheme not in ("http", "https"):
                        continue
                    # fragment removal
                    clean = parsed._replace(fragment="").geturl()
                    if clean in self.visited:
                        continue
                    if not self._same_registrable_domain(clean):
                        self.skipped_count += 1
                        continue
                    self.to_visit.put((clean, depth + 1))

        return results