import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
HISTORY_PATH = ROOT_DIR / "output" / "history.json"


class SelectedTrend(BaseModel):
    topic: str = Field(description="Concise, punchy YouTube Shorts topic title for US Tech/AI audience")
    reason: str = Field(description="Why this topic has high viral potential this week")


def fetch_hackernews_top(limit: int = 20) -> List[str]:
    """Fetch top story titles from HackerNews public API."""
    titles: List[str] = []
    try:
        ids = requests.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json", timeout=10
        ).json()[:limit]
        for item_id in ids:
            item = requests.get(
                f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json", timeout=5
            ).json()
            if item and item.get("title"):
                titles.append(f"[HN] {item['title']}")
    except Exception as exc:
        print(f"[TrendHunter] HackerNews fetch warning: {exc}")
    return titles


def fetch_reddit_top(subreddit: str, limit: int = 15) -> List[str]:
    """Fetch top weekly posts from a subreddit JSON feed."""
    titles: List[str] = []
    url = f"https://www.reddit.com/r/{subreddit}/top.json?t=week&limit={limit}"
    headers = {"User-Agent": "AGV-TrendHunter/1.0 (automated-shorts-factory)"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            posts = resp.json().get("data", {}).get("children", [])
            for p in posts:
                title = p.get("data", {}).get("title")
                if title:
                    titles.append(f"[r/{subreddit}] {title}")
    except Exception as exc:
        print(f"[TrendHunter] Reddit r/{subreddit} fetch warning: {exc}")
    return titles


def load_history() -> List[Dict[str, Any]]:
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_history(history: List[Dict[str, Any]]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def hunt_trending_topic() -> str:
    """Fetch trends, filter with gemini-2.5-flash, avoid history duplicates, and return topic."""
    candidates: List[str] = []
    candidates.extend(fetch_hackernews_top(20))
    candidates.extend(fetch_reddit_top("artificial", 15))
    candidates.extend(fetch_reddit_top("singularity", 15))

    if not candidates:
        raise RuntimeError("No trending topics could be fetched from HN or Reddit.")

    history = load_history()
    past_topics = [h["topic"] for h in history if "topic" in h]

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    prompt = f"""
You are a viral YouTube Shorts strategist targeting US Tech & AI audiences.

Here are this week's top discussions from HackerNews, r/artificial, and r/singularity:
{chr(10).join(candidates)}

Previously covered topics (DO NOT repeat or closely duplicate any of these):
{json.dumps(past_topics, indent=2)}

Task:
Filter out noise, memes, and low-impact chatter. Pick the SINGLE highest-viral-potential topic for a 45-second US Tech/AI YouTube Short.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SelectedTrend,
            temperature=0.4,
        ),
    )
    selected = json.loads(response.text)
    topic = selected["topic"].strip()

    history.append(
        {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "topic": topic,
            "reason": selected.get("reason", ""),
        }
    )
    save_history(history)
    print(f"[TrendHunter] Selected topic: {topic}")
    return topic


if __name__ == "__main__":
    hunt_trending_topic()
