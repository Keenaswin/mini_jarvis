"""
retrieval.py — Mini Jarvis Semantic Retrieval
===============================================
Handles the full retrieval pipeline:

  Query → Embed → FAISS cosine search → Fetch SQLite rows
       → Re-rank by importance → Format context for LLM

This module intentionally keeps search logic separate from
storage (memory.py) to make each component independently testable.
"""

import sqlite3
from typing import Dict, List, Optional

import numpy as np

from memory import MemoryStore
from utils import truncate_to_token_budget, cprint, Colors

# Default number of results to pull from FAISS before re-ranking
DEFAULT_TOP_K = 5


class Retriever:
    """
    Semantic retrieval engine backed by a MemoryStore.

    Usage:
        retriever = Retriever(store)
        results   = retriever.search("machine learning notes", top_k=5)
        context   = retriever.build_context(results)
    """

    def __init__(self, store: MemoryStore):
        """
        Args:
            store: An initialized MemoryStore instance (shared across modules).
        """
        self.store = store

    # ─────────────────────────────────────────
    # Core search
    # ─────────────────────────────────────────
    def search(
        self,
        query: str,
        top_k: int                      = DEFAULT_TOP_K,
        mem_type: Optional[str]         = None,
        min_score: float                = 0.10,
        boost_by_importance: bool       = True,
    ) -> List[Dict]:
        """
        Semantic search over stored memories.

        Steps:
          1. Embed the query
          2. FAISS inner-product search (equivalent to cosine on normalised vecs)
          3. Fetch matching rows from SQLite
          4. Optional type filter
          5. Optional importance re-ranking
          6. Return sorted results

        Args:
            query:                Natural-language query string.
            top_k:                Maximum number of results to return.
            mem_type:             Filter to 'note', 'task', or 'conversation'. None = all.
            min_score:            Minimum cosine similarity threshold (0–1).
            boost_by_importance:  Multiply cosine score by importance weight.

        Returns:
            List of memory dicts, each augmented with a 'score' field,
            sorted by score descending.
        """
        index = self.store.index

        if index.ntotal == 0:
            return []

        # ── 1. Embed query ─────────────────────────────────────────
        query_vec = self.store.embed_query(query)   # shape (1, 384)

        # ── 2. FAISS search ────────────────────────────────────────
        # Pull extra candidates before filtering/reranking
        fetch_k = min(top_k * 3, index.ntotal)
        scores, indices = index.search(query_vec, fetch_k)

        # ── 3. Fetch rows + filter ─────────────────────────────────
        results: List[Dict] = []

        conn = sqlite3.connect(self.store.db_path)
        conn.row_factory = sqlite3.Row

        for raw_score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue  # FAISS padding

            # Cosine similarity is in [0, 1] for normalized vectors
            cosine = float(raw_score)
            if cosine < min_score:
                continue

            # Map FAISS positional index → memory ID → SQLite row
            if idx >= len(self.store.index_meta):
                continue
            mem_id = self.store.index_meta[idx]

            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (mem_id,)
            ).fetchone()
            if row is None:
                continue

            mem = dict(row)

            # ── Type filter ────────────────────────────────────────
            if mem_type and mem["type"] != mem_type:
                continue

            # ── Importance boost ───────────────────────────────────
            final_score = cosine * mem["importance"] if boost_by_importance else cosine
            mem["score"]        = round(final_score, 4)
            mem["cosine_score"] = round(cosine, 4)

            results.append(mem)

        conn.close()

        # ── 4. Sort by combined score, keep top_k ─────────────────
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # ─────────────────────────────────────────
    # Context building for LLM prompt
    # ─────────────────────────────────────────
    def build_context(
        self,
        memories: List[Dict],
        max_tokens: int   = 1200,
        include_meta: bool = True,
    ) -> str:
        """
        Assemble retrieved memories into a context block suitable for
        injection into an LLM prompt.

        Memories are listed in order, each preceded by its type and
        timestamp so the model can reason about recency.

        Args:
            memories:     List of memory dicts (as returned by search()).
            max_tokens:   Hard limit on total context length.
            include_meta: If True, prefix each entry with [TYPE @ TIME].

        Returns:
            A formatted string block, or '' if memories is empty.
        """
        if not memories:
            return ""

        parts: List[str] = []
        for mem in memories:
            if include_meta:
                prefix = f"[{mem['type'].upper()} @ {mem['timestamp']}]"
                parts.append(f"{prefix}\n{mem['content']}")
            else:
                parts.append(mem["content"])

        combined = "\n\n---\n\n".join(parts)
        return truncate_to_token_budget(combined, max_tokens)

    # ─────────────────────────────────────────
    # Convenience: search + build in one call
    # ─────────────────────────────────────────
    def retrieve_context(
        self,
        query: str,
        top_k: int            = DEFAULT_TOP_K,
        mem_type: Optional[str] = None,
        max_tokens: int       = 1200,
    ) -> str:
        """
        One-liner: search → format context string for LLM.

        Returns '' if nothing is found above the similarity threshold.
        """
        memories = self.search(query, top_k=top_k, mem_type=mem_type)
        return self.build_context(memories, max_tokens=max_tokens)

    # ─────────────────────────────────────────
    # Debug / exploration
    # ─────────────────────────────────────────
    def print_results(self, results: List[Dict]) -> None:
        """Pretty-print search results to the terminal."""
        if not results:
            cprint("  No relevant memories found.", Colors.YELLOW)
            return

        cprint(f"\n  Found {len(results)} relevant memory/memories:\n", Colors.CYAN, bold=True)
        for i, mem in enumerate(results, 1):
            score_bar = "█" * int(mem["score"] * 20)
            cprint(
                f"  {i}. [{mem['type'].upper()}] score={mem['score']:.3f}  {score_bar}",
                Colors.BLUE,
                bold=True,
            )
            print(f"     {mem['timestamp']}  |  tags: {mem.get('tags','')}")
            preview = mem["content"][:200] + ("…" if len(mem["content"]) > 200 else "")
            print(f"     {preview}\n")
