"""
utils.py — Mini Jarvis Helper Utilities
========================================
Shared helper functions used across all modules:
  - Text chunking with overlap
  - Unique ID generation
  - Timestamp formatting
  - Natural-language datetime parsing
  - Token-budget text truncation
  - ANSI terminal color codes for CLI
"""

import re
import hashlib
from datetime import datetime, timedelta
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────
# ANSI color codes for pretty CLI output
# ─────────────────────────────────────────────
class Colors:
    HEADER    = "\033[95m"
    BLUE      = "\033[94m"
    CYAN      = "\033[96m"
    GREEN     = "\033[92m"
    YELLOW    = "\033[93m"
    RED       = "\033[91m"
    BOLD      = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET     = "\033[0m"

def cprint(text: str, color: str = Colors.RESET, bold: bool = False) -> None:
    """Print colored text to stdout."""
    prefix = Colors.BOLD if bold else ""
    print(f"{prefix}{color}{text}{Colors.RESET}")


# ─────────────────────────────────────────────
# Text Processing
# ─────────────────────────────────────────────
def chunk_text(
    text: str,
    chunk_size: int = 200,   # words per chunk
    overlap: int = 40        # overlapping words between chunks
) -> List[str]:
    """
    Split a long text into smaller overlapping chunks.

    Overlap ensures that context near chunk boundaries is
    captured in adjacent chunks — important for retrieval quality.

    Args:
        text:       The full text to split.
        chunk_size: Maximum number of words per chunk.
        overlap:    Number of words shared between consecutive chunks.

    Returns:
        A list of text chunks (strings).
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text.strip()]

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk.strip())
        if end == len(words):
            break
        start += chunk_size - overlap  # slide window with overlap

    return chunks


def truncate_to_token_budget(text: str, max_tokens: int = 800) -> str:
    """
    Rough token-budget truncation (approx. 4 chars ≈ 1 token).
    Used to keep LLM context within limits.

    Args:
        text:       The text to truncate.
        max_tokens: Approximate token limit.

    Returns:
        Truncated string, possibly with '…' appended.
    """
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " …"


# ─────────────────────────────────────────────
# ID & Timestamp Utilities
# ─────────────────────────────────────────────
def generate_id(content: str, timestamp: str) -> str:
    """
    Generate a short deterministic ID from content + timestamp.
    Uses MD5 for speed (not security — collisions are acceptable here).
    """
    raw = f"{content.strip()}{timestamp}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def now_str() -> str:
    """Return current local time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_datetime_str(dt_str: str) -> Optional[datetime]:
    """
    Parse a stored timestamp string back into a datetime object.

    Args:
        dt_str: Timestamp string in 'YYYY-MM-DD HH:MM:SS' format.

    Returns:
        datetime object, or None on failure.
    """
    try:
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────
# Natural Language Datetime Extraction
# ─────────────────────────────────────────────
def extract_due_time(text: str) -> Optional[datetime]:
    """
    Extract a due datetime from a natural-language string.

    Supports patterns like:
      - "tomorrow at 5 PM"
      - "today at 2:30 am"
      - "in 30 minutes"
      - "in 3 hours"
      - "in 2 days"

    Falls back to dateparser library if installed.

    Args:
        text: Natural language text (e.g. a task description).

    Returns:
        datetime if a time expression is found, else None.
    """
    text_lower = text.lower()

    # ── "tomorrow at HH[:MM] [am/pm]" ──────────────────────────────
    m = re.search(
        r'\btomorrow\b.*?\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?',
        text_lower
    )
    if m:
        return _build_time(datetime.now() + timedelta(days=1), m)

    # ── "today at HH[:MM] [am/pm]" ─────────────────────────────────
    m = re.search(
        r'\btoday\b.*?\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?',
        text_lower
    )
    if m:
        return _build_time(datetime.now(), m)

    # ── "in N minutes/hours/days" ───────────────────────────────────
    m = re.search(r'\bin\s+(\d+)\s+(minute|hour|day)s?', text_lower)
    if m:
        amount = int(m.group(1))
        unit   = m.group(2)
        delta  = {"minute": timedelta(minutes=amount),
                  "hour":   timedelta(hours=amount),
                  "day":    timedelta(days=amount)}.get(unit, timedelta())
        return datetime.now() + delta

    # ── Fallback: try dateparser if available ───────────────────────
    try:
        import dateparser  # type: ignore
        result = dateparser.parse(text, settings={"PREFER_DATES_FROM": "future"})
        return result
    except ImportError:
        pass

    return None


def _build_time(base: datetime, match: re.Match) -> datetime:
    """
    Build a concrete datetime from a regex match of HH[:MM] [am|pm].

    Args:
        base:  The base date (e.g. today or tomorrow).
        match: Regex match with groups (hour, minute, ampm).

    Returns:
        datetime with the extracted time applied to base.
    """
    hour   = int(match.group(1))
    minute = int(match.group(2) or 0)
    ampm   = match.group(3)

    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0

    return base.replace(hour=hour, minute=minute, second=0, microsecond=0)


# ─────────────────────────────────────────────
# Importance / Priority Scoring
# ─────────────────────────────────────────────
_PRIORITY_KEYWORDS = {
    "urgent":    3.0,
    "important": 2.5,
    "critical":  3.0,
    "asap":      2.8,
    "deadline":  2.5,
    "must":      2.0,
    "remember":  1.5,
    "note":      1.0,
    "remind":    1.5,
}

def score_importance(text: str) -> float:
    """
    Heuristic importance score for a memory entry (1.0 = normal).
    Higher scores surface this memory higher in search results.

    Args:
        text: The memory content.

    Returns:
        Float importance score (1.0 – 3.0).
    """
    text_lower = text.lower()
    score = 1.0
    for keyword, boost in _PRIORITY_KEYWORDS.items():
        if keyword in text_lower:
            score = max(score, boost)
    return round(score, 2)


# ─────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────
def format_memory_for_display(mem: dict) -> str:
    """Format a memory dict into a human-readable string for the CLI."""
    ts   = mem.get("timestamp", "")
    mtype = mem.get("type", "note").upper()
    score = mem.get("score", None)
    score_str = f"  [score={score:.3f}]" if score is not None else ""
    return (
        f"  [{mtype}] {ts}{score_str}\n"
        f"  {mem['content'][:200]}{'…' if len(mem['content']) > 200 else ''}"
    )
