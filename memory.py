"""
memory.py — Mini Jarvis Memory System
=======================================
Manages the core memory pipeline:

  1. Accept raw text (notes, tasks, conversations)
  2. Chunk long text into overlapping segments
  3. Embed each chunk with sentence-transformers
  4. Persist raw text → SQLite
  5. Persist embeddings → FAISS vector index

Supports:
  - add_memory()      → store + embed new entry
  - delete_memory()   → remove by ID, rebuild index
  - update_memory()   → change content, re-embed
  - get_all_memories()→ list by type
  - search() is in retrieval.py (keeps concerns separated)
"""

import os
import pickle
import sqlite3
from typing import Dict, List, Optional

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore

from utils import (
    chunk_text,
    generate_id,
    now_str,
    score_importance,
    cprint,
    Colors,
)

# ─────────────────────────────────────────────
# Constants — adjust paths to taste
# ─────────────────────────────────────────────
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   # ~80 MB, 384-dim, fast on CPU
EMBEDDING_DIM    = 384
DB_PATH          = "minijarvis.db"
FAISS_INDEX_PATH = "minijarvis.faiss"
FAISS_META_PATH  = "minijarvis_meta.pkl"

VALID_TYPES = {"note", "task", "conversation"}


# ─────────────────────────────────────────────
# Database helpers (kept local to this module)
# ─────────────────────────────────────────────
def _get_conn(db_path: str) -> sqlite3.Connection:
    """Return a SQLite connection with row_factory set."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    """
    Create all required tables if they don't already exist.

    Schema:
      memories       — raw text chunks + metadata
      conversations  — full turn-by-turn chat history
      tasks          — scheduled reminders
    """
    c = conn.cursor()

    # Core memory table
    c.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id           TEXT PRIMARY KEY,
            content      TEXT    NOT NULL,
            chunk_index  INTEGER DEFAULT 0,
            parent_id    TEXT,
            timestamp    TEXT    NOT NULL,
            type         TEXT    NOT NULL,
            importance   REAL    DEFAULT 1.0,
            tags         TEXT    DEFAULT ''
        )
    """)

    # Conversation history
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            role       TEXT NOT NULL,      -- 'user' | 'assistant' | 'system'
            content    TEXT NOT NULL,
            timestamp  TEXT NOT NULL,
            session_id TEXT DEFAULT 'default'
        )
    """)

    # Task / reminder table
    c.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            content    TEXT NOT NULL,
            due_time   TEXT,               -- ISO-ish: '2025-01-01 17:00:00'
            created_at TEXT NOT NULL,
            completed  INTEGER DEFAULT 0,  -- 0 = pending, 1 = done
            reminded   INTEGER DEFAULT 0   -- 0 = not yet notified
        )
    """)

    conn.commit()


# ─────────────────────────────────────────────
# MemoryStore class
# ─────────────────────────────────────────────
class MemoryStore:
    """
    Unified store for memories.

    Internally manages:
      - An SQLite database for raw text and metadata
      - A FAISS flat index for semantic (cosine) similarity search

    Usage:
        store = MemoryStore()
        ids   = store.add_memory("Buy milk", mem_type="note")
        mems  = store.get_all_memories(mem_type="note")
        store.delete_memory(ids[0])
    """

    def __init__(
        self,
        db_path: str          = DB_PATH,
        faiss_path: str       = FAISS_INDEX_PATH,
        meta_path: str        = FAISS_META_PATH,
        model_name: str       = EMBEDDING_MODEL,
    ):
        self.db_path    = db_path
        self.faiss_path = faiss_path
        self.meta_path  = meta_path

        # ── Load embedding model ───────────────────────────────────
        cprint("⚙  Loading embedding model (first run downloads ~80 MB)…", Colors.CYAN)
        self.encoder = SentenceTransformer(model_name)
        cprint(f"✓  Embedding model ready: {model_name}", Colors.GREEN)

        # ── SQLite ─────────────────────────────────────────────────
        conn = _get_conn(self.db_path)
        _init_db(conn)
        conn.close()

        # ── FAISS index ────────────────────────────────────────────
        self._load_or_create_index()

    # ─────────────────────────────────────────
    # FAISS index lifecycle
    # ─────────────────────────────────────────
    def _load_or_create_index(self) -> None:
        """Load a persisted FAISS index, or create a fresh one."""
        if (
            os.path.exists(self.faiss_path)
            and os.path.exists(self.meta_path)
        ):
            self.index = faiss.read_index(self.faiss_path)
            with open(self.meta_path, "rb") as f:
                # index_meta: list of memory IDs, positionally aligned with FAISS
                self.index_meta: List[str] = pickle.load(f)
            cprint(
                f"✓  FAISS index loaded ({self.index.ntotal} vectors)",
                Colors.GREEN,
            )
        else:
            # IndexFlatIP with normalized vectors = cosine similarity
            self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
            self.index_meta: List[str] = []
            cprint("✓  New FAISS index created", Colors.GREEN)

    def _save_index(self) -> None:
        """Persist FAISS index and its metadata mapping to disk."""
        faiss.write_index(self.index, self.faiss_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.index_meta, f)

    def _rebuild_index(self) -> None:
        """
        Fully rebuild the FAISS index from scratch using SQLite.

        Required after deletes because FAISS FlatIP doesn't support
        in-place removal of individual vectors.
        """
        conn = _get_conn(self.db_path)
        rows = conn.execute("SELECT id, content FROM memories").fetchall()
        conn.close()

        # Reset index
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index_meta = []

        if not rows:
            self._save_index()
            return

        contents   = [r["content"] for r in rows]
        ids        = [r["id"]      for r in rows]
        embeddings = self.encoder.encode(
            contents, normalize_embeddings=True, show_progress_bar=False
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings)
        self.index_meta = ids
        self._save_index()
        cprint(f"✓  FAISS index rebuilt ({self.index.ntotal} vectors)", Colors.GREEN)

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────
    def add_memory(
        self,
        content: str,
        mem_type: str   = "note",
        importance: float = None,
        tags: str        = "",
    ) -> List[str]:
        """
        Store a new memory entry.

        Steps:
          1. Validate type
          2. Auto-score importance if not supplied
          3. Chunk the content
          4. Embed each chunk
          5. Insert into SQLite
          6. Add embedding to FAISS
          7. Persist index

        Args:
            content:    The text to store.
            mem_type:   One of 'note', 'task', 'conversation'.
            importance: Override auto-scored importance (1.0 – 3.0).
            tags:       Comma-separated tag string, e.g. "work,project-x".

        Returns:
            List of memory IDs created (one per chunk).
        """
        if mem_type not in VALID_TYPES:
            raise ValueError(f"mem_type must be one of {VALID_TYPES}")

        timestamp  = now_str()
        importance = importance if importance is not None else score_importance(content)
        chunks     = chunk_text(content)
        parent_id  = generate_id(content, timestamp)
        added_ids: List[str] = []

        conn = _get_conn(self.db_path)
        try:
            for i, chunk in enumerate(chunks):
                # Unique chunk ID (parent_id for single-chunk entries)
                mem_id = f"{parent_id}_{i}" if len(chunks) > 1 else parent_id

                # ── Embed ──────────────────────────────────────────
                embedding = self.encoder.encode(
                    [chunk], normalize_embeddings=True
                )[0]
                embedding_np = np.array([embedding], dtype=np.float32)

                # ── SQLite ─────────────────────────────────────────
                conn.execute(
                    """
                    INSERT OR REPLACE INTO memories
                        (id, content, chunk_index, parent_id, timestamp, type, importance, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (mem_id, chunk, i, parent_id, timestamp, mem_type, importance, tags),
                )

                # ── FAISS ──────────────────────────────────────────
                self.index.add(embedding_np)
                self.index_meta.append(mem_id)
                added_ids.append(mem_id)

            conn.commit()
        finally:
            conn.close()

        self._save_index()
        return added_ids

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string for use in FAISS search.

        Returns:
            Normalized float32 embedding of shape (1, EMBEDDING_DIM).
        """
        vec = self.encoder.encode([query], normalize_embeddings=True)[0]
        return np.array([vec], dtype=np.float32)

    def get_vectors_for_ids(self, ids: List[str]) -> List[Optional[int]]:
        """
        Return FAISS positional indices for a list of memory IDs.
        Used internally by retrieval.py.
        """
        return [
            self.index_meta.index(mid) if mid in self.index_meta else None
            for mid in ids
        ]

    def get_memory_by_id(self, mem_id: str) -> Optional[Dict]:
        """Fetch a single memory record by its ID."""
        conn = _get_conn(self.db_path)
        row  = conn.execute(
            "SELECT * FROM memories WHERE id = ?", (mem_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_all_memories(self, mem_type: Optional[str] = None) -> List[Dict]:
        """
        Retrieve all stored memories, optionally filtered by type.

        Args:
            mem_type: Filter by 'note', 'task', or 'conversation'. None = all.

        Returns:
            List of memory dicts, most recent first.
        """
        conn = _get_conn(self.db_path)
        if mem_type:
            rows = conn.execute(
                "SELECT * FROM memories WHERE type = ? ORDER BY timestamp DESC",
                (mem_type,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM memories ORDER BY timestamp DESC"
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def delete_memory(self, mem_id: str) -> bool:
        """
        Delete a memory (and all its chunks) by parent ID or chunk ID.

        After deletion the FAISS index is rebuilt to keep it consistent.

        Args:
            mem_id: The ID string of the memory to remove.

        Returns:
            True if any rows were deleted, False otherwise.
        """
        conn = _get_conn(self.db_path)
        cursor = conn.execute(
            "DELETE FROM memories WHERE id = ? OR parent_id = ?",
            (mem_id, mem_id),
        )
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()

        if deleted:
            self._rebuild_index()

        return deleted

    def update_memory(self, mem_id: str, new_content: str) -> bool:
        """
        Update the text of an existing memory and re-embed it.

        Args:
            mem_id:      ID of the memory chunk to update.
            new_content: Replacement text.

        Returns:
            True if the record existed and was updated.
        """
        conn = _get_conn(self.db_path)
        cursor = conn.execute(
            "UPDATE memories SET content = ? WHERE id = ?",
            (new_content, mem_id),
        )
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()

        if updated:
            self._rebuild_index()

        return updated

    def stats(self) -> Dict:
        """Return a summary of what's stored."""
        conn = _get_conn(self.db_path)
        total  = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        notes  = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE type='note'"
        ).fetchone()[0]
        tasks  = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE type='task'"
        ).fetchone()[0]
        convos = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE type='conversation'"
        ).fetchone()[0]
        conn.close()
        return {
            "total":        total,
            "notes":        notes,
            "tasks":        tasks,
            "conversations": convos,
            "faiss_vectors": self.index.ntotal,
        }
