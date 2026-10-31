import os
import re
from datetime import datetime
from urllib.parse import urlparse
from agents import build_search_agent, build_reader_agent, planner_chain, summarizer_chain, fact_chain, writer_chain, revision_writer_chain, critic_chain

DEPTH = {
    "quick": {"questions": 3, "pages": 3},
    "standard": {"questions": 4, "pages": 5},
    "deep": {"questions": 6, "pages": 8}
}

MAX_SUMMARY_INPUT = 4500
MAX_FACT_SOURCE_INPUT = 4500

def header(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)

def parse_questions(text: str, n: int):
    questions = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:\d+[\.\)\-]\s*|[-*]\s*)", "", line).strip()
        if line:
            questions.append(line)
    return questions[:n]

def get_tool_output(result: dict, expected_tool: str = None) -> str:
    for m in reversed(result["messages"]):
        if m.type != "tool" or (expected_tool and getattr(m, "name", None) != expected_tool):
            continue
        if isinstance(m.content, str):
            return m.content
        if isinstance(m.content, list):
            return "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in m.content
            )
        return str(m.content)
    return ""

def is_valid_article(text: str) -> bool:
    if not text or len(text.strip()) < 300:
        return False
    bad = [
        "enable javascript",
        "javascript is disabled",
        "access denied",
        "checking your browser",
        "captcha",
        "please enable cookies",
        "error scraping url:",
        "no readable webpage content found",
        "pdf contains no extractable text"
    ]
    text_lower = text.lower()
    return not any(item in text_lower for item in bad)

def select_sources(results: list, limit: int) -> list:
    selected = []
    remaining = []
    domain_counts = {}
    for item in sorted(results, key=lambda x: x.get("score", 0), reverse=True):
        domain = item.get("source") or urlparse(item.get("url", "")).hostname or "unknown"
        if domain_counts.get(domain, 0) < 2 and len(selected) < limit:
            selected.append(item)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        else:
            remaining.append(item)
    for item in remaining:
        if len(selected) >= limit:
            break
        if item not in selected:
            selected.append(item)
    return selected

def build_references(sources: list) -> str:
    references = []
    for s in sources:
        references.append(f"[{s['id']}] [{s['title']}]({s['url']})")
    return "\n".join(references)

def build_source_evidence(sources: list, max_length: int = MAX_FACT_SOURCE_INPUT) -> str:
    return "\n\n".join(
        f"[{s['id']}] {s['title']}\n"
        f"URL: {s['url']}\n"
        f"Publication Date: {s['published_date'] or 'Not available'}\n"
        f"Evidence Type: {s['evidence_type']}\n"
        f"Original Evidence:\n{s['article_text'][:max_length]}"
        for s in sources
    )

def save_report(state: dict) -> str:
    os.makedirs("reports", exist_ok=True)
    name = re.sub(r"[^a-zA-Z0-9]+", "_", state["topic"]).strip("_")[:40] or "research_report"
    path = f"reports/{name}_{datetime.now():%Y%m%d_%H%M%S}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(state["report"])
        f.write("\n\n---\n\n## Initial Draft\n")
        f.write(state["draft_report"])
        f.write("\n\n---\n\n## Fact Check\n")
        f.write(state["facts"])
        f.write("\n\n---\n\n## Critic Review\n")
        f.write(state["feedback"])
    return path

def run_research_pipeline(topic: str, depth: str = "standard") -> dict:
    cfg = DEPTH.get(depth, DEPTH["standard"])
    state = {
        "topic": topic,
        "depth": depth
    }

    header("STEP 1 -> Research Planner is Executing")
    plan = planner_chain.invoke({"topic": topic, "n": cfg["questions"]})
    state["questions"] = parse_questions(plan, cfg["questions"])
    if not state["questions"]:
        raise ValueError("Research Planner did not generate any questions.")
    print("\nResearch Questions:")
    for i, q in enumerate(state["questions"], 1):
        print(f"{i}. {q}")

    header("STEP 2 -> Search Agent is Executing")
    search_agent = build_search_agent()
    search_notes = []
    all_results = []
    seen = set()
    for q in state["questions"]:
        print(f"\nSearching: {q}")
        results = []
        try:
            search_result = search_agent.invoke({
                "messages": [("user", f"Find recent, reliable and relevant sources that directly answer this research question: {q}")]
            })
            for m in search_result["messages"]:
                if m.type == "tool" and getattr(m, "name", None) == "tavily_search":
                    artifact = getattr(m, "artifact", None)
                    if isinstance(artifact, list):
                        results.extend(artifact)
        except Exception as e:
            print(f"Search Agent failed for '{q}': {e}")
        if not results:
            print("Search Agent returned no results for this question.")
            continue
        search_notes.append(
            f"Question: {q}\n" +
            "\n".join(
                f"{r.get('title', 'Untitled')}: {r.get('url', '')}"
                for r in results
            )
        )
        for item in results:
            url = item.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)
            all_results.append(item)

    selected_results = select_sources(all_results, cfg["pages"])
    sources = []
    for item in selected_results:
        sources.append({
            "id": len(sources) + 1,
            "title": item.get("title", "Untitled Source"),
            "url": item["url"],
            "content": item.get("content", ""),
            "score": item.get("score", 0),
            "published_date": item.get("published_date", ""),
            "source": item.get("source", "")
        })

    state["sources"] = sources
    state["search_results"] = "\n\n".join(search_notes)
    if not sources:
        raise ValueError("No sources found. Check your Tavily API key and Search Agent.")

    print("\nURLs Found:")
    for s in sources:
        print(f"\n[{s['id']}] {s['title']}")
        print(f"URL: {s['url']}")
        print(f"Domain: {s['source']}")
        print(f"Published Date: {s['published_date'] or 'Not available'}")
        print(f"Relevance Score: {s['score']}")

    header("STEP 3 -> Reader Agent is Executing")
    reader_agent = build_reader_agent()
    for s in sources:
        print(f"\nReading Source [{s['id']}]: {s['title']}")
        try:
            reader_result = reader_agent.invoke({
                "messages": [("user", f"Use the scrape_url tool to extract the content of this URL: {s['url']}")]
            })
            article = get_tool_output(reader_result, expected_tool="scrape_url")
            if is_valid_article(article):
                s["article_text"] = article
                s["evidence_type"] = "Extracted PDF" if article.startswith("[PDF DOCUMENT]") else "Extracted Webpage"
                print(f"{s['evidence_type']} extracted successfully.")
            else:
                s["article_text"] = s.get("content", "")
                s["evidence_type"] = "Tavily Search Snippet" if s["article_text"] else "No Evidence"
                print("Extraction failed. Using Tavily search snippet.")
        except Exception as e:
            s["article_text"] = s.get("content", "")
            s["evidence_type"] = "Tavily Search Snippet" if s["article_text"] else "No Evidence"
            print(f"Reader Agent failed: {e}")
        if not s["article_text"]:
            s["article_text"] = "No readable evidence available."
            s["evidence_type"] = "No Evidence"

    header("STEP 4 -> Article Summarizer is Executing")
    for source in sources:
        print(f"\nSummarizing Source [{source['id']}]: {source['title']}")
        article = source.get("article_text", "")
        if not article or article == "No readable evidence available.":
            source["summary"] = "No readable article content available."
            continue
        try:
            source["summary"] = summarizer_chain.invoke({
                "article": article[:MAX_SUMMARY_INPUT]
            })
        except Exception as e:
            print(f"Summarization failed for source [{source['id']}]: {e}")
            source["summary"] = "Summary unavailable."

    state["reader_results"] = "\n\n".join(
        f"[{source['id']}] {source['title']}\n"
        f"URL: {source['url']}\n"
        f"Publication Date: {source['published_date'] or 'Not available'}\n"
        f"Evidence Type: {source['evidence_type']}\n"
        f"Summary:\n{source['summary']}"
        for source in sources
    )
    print("\nSummarized Research:")
    print(state["reader_results"])

    header("STEP 5 -> Writer is Creating Initial Report")
    state["draft_report"] = writer_chain.invoke({
        "topic": topic,
        "questions": "\n".join(state["questions"]),
        "research": state["reader_results"]
    })
    print("\nInitial Research Report:")
    print(state["draft_report"])

    header("STEP 6 -> Fact Checker is Verifying Initial Report")
    fact_sources = build_source_evidence(sources)
    state["facts"] = fact_chain.invoke({
        "report": state["draft_report"],
        "sources": fact_sources
    })
    print("\nFact Check Results:")
    print(state["facts"])

    header("STEP 7 -> Writer is Revising the Report")
    revised_body = revision_writer_chain.invoke({
        "topic": topic,
        "draft": state["draft_report"],
        "facts": state["facts"],
        "research": state["reader_results"]
    })
    state["report"] = (
        revised_body
        + "\n\n## References\n"
        + build_references(sources)
    )
    print("\nRevised Research Report:")
    print(state["report"])

    header("STEP 8 -> Critic is Reviewing")
    state["feedback"] = critic_chain.invoke({
        "report": state["report"],
        "source_count": len(sources),
        "facts": state["facts"],
        "sources": fact_sources
    })
    print("\nCritic Review:")
    print(state["feedback"])

    header("STEP 9 -> Saving Research Report")
    state["file"] = save_report(state)
    print("\nReport Saved Successfully!")
    print("Location:", state["file"])
    return state

if __name__ == "__main__":
    topic = input("\nEnter a research topic: ").strip()
    if not topic:
        raise ValueError("Research topic cannot be empty.")
    depth = input("Depth (quick / standard / deep) [standard]: ").strip().lower() or "standard"
    if depth not in DEPTH:
        print("Invalid depth. Using standard.")
        depth = "standard"
    run_research_pipeline(topic, depth)