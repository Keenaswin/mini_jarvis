"""
conversation.py — Mini Jarvis Conversation Manager
====================================================
Manages the multi-turn chat session:

  - Persists every turn to SQLite (role, content, timestamp, session_id)
  - Maintains an in-memory turn buffer for fast access
  - Handles context-window overflow by summarising old turns
  - Provides helpers for injecting history into the LLM prompt

Design:
  A "session" is a continuous conversation sequence identified by a
  session_id string (defaults to "default").  Each turn is saved
  permanently so history survives restarts, but only the most recent
  turns (within the token budget) are used in LLM prompts.
"""

import sqlite3
import textwrap
from datetime import datetime
from typing import Dict, List, Optional

from utils import now_str, truncate_to_token_budget, cprint, Colors

# Maximum turns kept in the in-memory buffer before summarisation
MAX_BUFFER_TURNS   = 40
# Maximum turns passed verbatim to the LLM per request
MAX_PROMPT_TURNS   = 12
# Approximate tokens per turn (generous estimate)
TOKENS_PER_TURN    = 150


class ConversationManager:
    """
    Manages chat history storage and context window handling.

    Usage:
        conv = ConversationManager(db_path="minijarvis.db")
        conv.add_turn("user",      "What did I write about Python?")
        conv.add_turn("assistant", "You wrote that Python is awesome.")
        history = conv.get_recent_history(max_turns=8)
    """

    def __init__(
        self,
        db_path: str     = "minijarvis.db",
        session_id: str  = "default",
    ):
        self.db_path    = db_path
        self.session_id = session_id
        # In-memory buffer for the active session
        self._buffer: List[Dict] = []

        self._ensure_table()
        self._load_session()

    # ─────────────────────────────────────────
    # Database bootstrap
    # ─────────────────────────────────────────
    def _ensure_table(self) -> None:
        """Create the conversations table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                timestamp  TEXT NOT NULL,
                session_id TEXT DEFAULT 'default'
            )
        """)
        conn.commit()
        conn.close()

    def _load_session(self) -> None:
        """Load the most recent turns of the active session into the buffer."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT role, content, timestamp
              FROM conversations
             WHERE session_id = ?
             ORDER BY id DESC
             LIMIT ?
            """,
            (self.session_id, MAX_BUFFER_TURNS),
        ).fetchall()
        conn.close()

        # Reverse to chronological order
        self._buffer = [dict(r) for r in reversed(rows)]

    # ─────────────────────────────────────────
    # Core turn management
    # ─────────────────────────────────────────
    def add_turn(self, role: str, content: str) -> None:
        """
        Persist a new conversation turn and append it to the buffer.

        Args:
            role:    'user', 'assistant', or 'system'.
            content: The message text.
        """
        timestamp = now_str()
        turn      = {"role": role, "content": content, "timestamp": timestamp}

        # ── Persist to SQLite ──────────────────────────────────────
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO conversations (role, content, timestamp, session_id) VALUES (?,?,?,?)",
            (role, content, timestamp, self.session_id),
        )
        conn.commit()
        conn.close()

        # ── Update buffer ──────────────────────────────────────────
        self._buffer.append(turn)

        # Trim buffer to MAX_BUFFER_TURNS
        if len(self._buffer) > MAX_BUFFER_TURNS:
            self._buffer = self._buffer[-MAX_BUFFER_TURNS:]

    # ─────────────────────────────────────────
    # History access
    # ─────────────────────────────────────────
    def get_recent_history(self, max_turns: int = MAX_PROMPT_TURNS) -> List[Dict]:
        """
        Return the most recent `max_turns` turns from the buffer.

        Args:
            max_turns: How many turns to include.

        Returns:
            List of {'role', 'content', 'timestamp'} dicts, oldest first.
        """
        return self._buffer[-max_turns:]

    def get_history_for_prompt(
        self,
        max_turns: int   = MAX_PROMPT_TURNS,
        max_tokens: int  = 800,
    ) -> List[Dict]:
        """
        Return history suitable for LLM prompt injection.

        Truncates individual turn content to keep within the token budget.

        Args:
            max_turns:  Maximum number of turns to include.
            max_tokens: Total approximate token budget for history.

        Returns:
            List of truncated turn dicts.
        """
        recent = self.get_recent_history(max_turns)
        budget = max_tokens
        result: List[Dict] = []

        # Walk backwards (newest first) to prioritise recent context
        for turn in reversed(recent):
            tokens_this_turn = len(turn["content"].split()) // 0.75  # rough estimate
            if budget <= 0:
                break
            truncated = truncate_to_token_budget(turn["content"], int(budget))
            result.insert(0, {**turn, "content": truncated})
            budget -= TOKENS_PER_TURN

        return result

    def get_all_sessions(self) -> List[str]:
        """Return a list of all unique session IDs."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT DISTINCT session_id FROM conversations"
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]

    def get_full_session_history(self, session_id: Optional[str] = None) -> List[Dict]:
        """
        Retrieve complete history for a session.

        Args:
            session_id: Target session. Defaults to the active session.

        Returns:
            Full ordered list of turns.
        """
        sid  = session_id or self.session_id
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT role, content, timestamp FROM conversations WHERE session_id = ? ORDER BY id",
            (sid,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ─────────────────────────────────────────
    # Summarisation
    # ─────────────────────────────────────────
    def summarise_old_history(self, llm, keep_recent: int = 10) -> Optional[str]:
        """
        Summarise history that won't fit in the prompt window.

        Uses the LLM to compress old turns into a single summary block.
        The summary is then stored as a 'system' turn so it can be
        included in future prompts without blowing the context budget.

        Args:
            llm:         A LocalLLM instance for generating the summary.
            keep_recent: Number of most-recent turns to leave untouched.

        Returns:
            The generated summary string, or None if nothing to summarise.
        """
        if len(self._buffer) <= keep_recent:
            return None  # Nothing to summarise

        old_turns   = self._buffer[:-keep_recent]
        recent_turns = self._buffer[-keep_recent:]

        # Format old turns for the LLM
        text_to_summarise = "\n".join(
            f"{t['role'].upper()}: {t['content']}" for t in old_turns
        )

        summary = llm.summarise(text_to_summarise)

        # Replace old turns in buffer with the summary
        summary_turn = {
            "role":      "system",
            "content":   f"[Summary of earlier conversation]\n{summary}",
            "timestamp": now_str(),
        }
        self._buffer = [summary_turn] + recent_turns

        # Persist summary turn to SQLite too
        self.add_turn("system", summary_turn["content"])
        return summary

    # ─────────────────────────────────────────
    # Session management
    # ─────────────────────────────────────────
    def new_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a fresh conversation session.

        Args:
            session_id: Custom session ID. Auto-generates a timestamped
                        ID if not provided.

        Returns:
            The new session_id string.
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.session_id = session_id
        self._buffer    = []
        cprint(f"✓  New session started: {session_id}", Colors.GREEN)
        return session_id

    def clear_session(self, session_id: Optional[str] = None) -> int:
        """
        Delete all turns in a session from SQLite.

        Args:
            session_id: Session to clear. Defaults to active session.

        Returns:
            Number of rows deleted.
        """
        sid  = session_id or self.session_id
        conn = sqlite3.connect(self.db_path)
        cur  = conn.execute(
            "DELETE FROM conversations WHERE session_id = ?", (sid,)
        )
        deleted = cur.rowcount
        conn.commit()
        conn.close()

        if sid == self.session_id:
            self._buffer = []

        cprint(f"✓  Cleared {deleted} turns from session '{sid}'.", Colors.GREEN)
        return deleted

    # ─────────────────────────────────────────
    # Display helpers
    # ─────────────────────────────────────────
    def print_history(self, max_turns: int = 20) -> None:
        """Print the recent conversation history to the terminal."""
        turns = self.get_recent_history(max_turns)
        if not turns:
            cprint("  (no conversation history in this session)", Colors.YELLOW)
            return

        cprint(f"\n  ── Conversation history (last {len(turns)} turns) ──\n", Colors.CYAN, bold=True)
        for turn in turns:
            role_color = Colors.BLUE if turn["role"] == "user" else Colors.GREEN
            cprint(f"  [{turn['role'].upper()}]  {turn['timestamp']}", role_color, bold=True)
            # Wrap long messages
            wrapped = textwrap.fill(turn["content"], width=80, initial_indent="  ", subsequent_indent="  ")
            print(wrapped)
            print()
