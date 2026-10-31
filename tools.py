from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import pymupdf
from urllib.parse import urlparse, urldefrag
import os

api_key = os.getenv("TAVILY_API_KEY")
if not api_key:
    raise ValueError("TAVILY_API_KEY is missing from .env file.")

tavily = TavilyClient(api_key=api_key)

def clean_url(url: str) -> str:
    return urldefrag(url)[0].rstrip("/")

def get_search_urls(query: str, max_results: int = 6) -> list:
    results = tavily.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        topic="general"
    )
    items = []
    for r in results.get("results", []):
        url = r.get("url", "")
        if not url:
            continue
        parsed = urlparse(url)
        items.append({
            "title": r.get("title", "Untitled Source"),
            "url": clean_url(url),
            "content": r.get("content", ""),
            "score": r.get("score", 0),
            "published_date": r.get("published_date", ""),
            "source": (parsed.hostname or "").removeprefix("www.")
        })
    return items

@tool(response_format="content_and_artifact")
def tavily_search(query: str):
    """Search the web and return relevant titles, URLs and snippets."""
    items = get_search_urls(query)
    out = []
    for r in items:
        out.append(
            f"Title: {r['title']}\n"
            f"URL: {r['url']}\n"
            f"Publication Date: {r['published_date'] or 'Not available'}\n"
            f"Snippet: {r['content'][:400]}\n"
        )
    return "\n---\n".join(out), items

@tool
def scrape_url(url: str) -> str:
    """Extract readable text from a webpage or PDF document."""
    try:
        response = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        is_pdf = (
            "application/pdf" in content_type
            or url.lower().split("?")[0].endswith(".pdf")
            or response.content.startswith(b"%PDF")
        )
        if is_pdf:
            with pymupdf.open(stream=response.content, filetype="pdf") as pdf_document:
                text = "\n".join(page.get_text() for page in pdf_document)
            if not text.strip():
                return "PDF contains no extractable text. It may be scanned."
            return "[PDF DOCUMENT]\n" + text[:12000]
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]):
            tag.decompose()
        article = soup.find("article") or soup.find("main") or soup.body or soup
        text = article.get_text(separator=" ", strip=True)
        if len(text.strip()) < 100:
            return "No readable webpage content found."
        return "[WEBPAGE]\n" + text[:12000]
    except Exception as e:
        return f"Error scraping URL: {str(e)}"