"""
llm.py — Mini Jarvis Local LLM Interface
==========================================
Wraps llama-cpp-python for fully offline LLM inference.

Key responsibilities:
  - Load a .gguf model file at startup
  - Accept a prompt (system + context + conversation + query)
  - Stream or return a response
  - Manage context-window limits (truncation / summarisation)
  - Provide a graceful "echo" fallback when no model is present

Supported model formats: any GGUF file compatible with llama-cpp-python.
Recommended small models (≤8 GB RAM):
  - Phi-3-mini-4k-instruct.Q4_K_M.gguf   (~2.3 GB)
  - Mistral-7B-Instruct-v0.3.Q4_K_M.gguf (~4.4 GB)
  - Llama-3.2-3B-Instruct.Q4_K_M.gguf    (~2.0 GB)
"""

import os
import textwrap
from typing import Iterator, List, Optional

from utils import truncate_to_token_budget, cprint, Colors

# ─────────────────────────────────────────────
# System prompt that shapes Jarvis's persona
# ─────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are Mini Jarvis, a helpful, concise, privacy-first personal AI assistant.
    You run entirely offline on the user's device — no data ever leaves their machine.

    Guidelines:
    - Use the provided memory context to answer questions accurately.
    - If you don't know something and it's not in the context, say so honestly.
    - Keep responses focused and actionable.
    - Cite relevant memories when they inform your answer.
    - Never fabricate dates, facts, or user data.
""").strip()


class LocalLLM:
    """
    Thin wrapper around llama-cpp-python's Llama class.

    Falls back to an echo/template mode when no model path is configured,
    so the rest of the system remains functional for testing.

    Usage:
        llm  = LocalLLM(model_path="models/phi3-mini.gguf")
        resp = llm.chat(user_query="What did I note about Python?",
                        context="[NOTE] You stored: Python is awesome")
        print(resp)
    """

    def __init__(
        self,
        model_path: Optional[str]  = None,
        n_ctx: int                 = 4096,    # context window in tokens
        n_threads: int             = 4,        # CPU threads
        n_gpu_layers: int          = 0,        # set >0 if you have GPU/Metal
        temperature: float         = 0.7,
        max_tokens: int            = 512,
        verbose: bool              = False,
    ):
        """
        Args:
            model_path:    Path to a .gguf model file.
                           If None or not found, runs in fallback mode.
            n_ctx:         Token context window size.
            n_threads:     CPU threads for inference.
            n_gpu_layers:  Layers to offload to GPU (0 = CPU only).
            temperature:   Sampling temperature (0 = deterministic).
            max_tokens:    Maximum tokens in the generated response.
            verbose:       Print llama.cpp internal logs.
        """
        self.model_path   = model_path
        self.n_ctx        = n_ctx
        self.n_threads    = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.temperature  = temperature
        self.max_tokens   = max_tokens
        self.verbose      = verbose
        self._llm         = None   # lazy-loaded Llama instance
        self._fallback    = True

        self._load_model()

    # ─────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────
    def _load_model(self) -> None:
        """
        Attempt to load the GGUF model via llama-cpp-python.
        Sets self._fallback = True if loading fails.
        """
        if not self.model_path:
            cprint(
                "⚠  No model_path set — running in FALLBACK mode (no real LLM).",
                Colors.YELLOW,
            )
            return

        if not os.path.exists(self.model_path):
            cprint(
                f"⚠  Model not found at '{self.model_path}' — running in FALLBACK mode.",
                Colors.YELLOW,
            )
            return

        try:
            from llama_cpp import Llama  # type: ignore

            cprint(f"⚙  Loading LLM from {self.model_path} …", Colors.CYAN)
            self._llm = Llama(
                model_path    = self.model_path,
                n_ctx         = self.n_ctx,
                n_threads     = self.n_threads,
                n_gpu_layers  = self.n_gpu_layers,
                verbose       = self.verbose,
            )
            self._fallback = False
            cprint("✓  Local LLM loaded and ready.", Colors.GREEN)

        except ImportError:
            cprint(
                "⚠  llama-cpp-python not installed. Run: pip install llama-cpp-python\n"
                "   Continuing in FALLBACK mode.",
                Colors.YELLOW,
            )
        except Exception as exc:
            cprint(f"⚠  Failed to load LLM: {exc}\n   Continuing in FALLBACK mode.", Colors.RED)

    # ─────────────────────────────────────────
    # Prompt construction
    # ─────────────────────────────────────────
    def _build_prompt(
        self,
        user_query: str,
        context: str,
        history: Optional[List[dict]] = None,
    ) -> str:
        """
        Assemble the full prompt sent to the model.

        Format (llama-2-chat / Mistral instruct style):
            [SYSTEM] <system_prompt>
            [MEMORY CONTEXT] <retrieved memories>
            [HISTORY] <recent conversation turns>
            [USER] <current query>
            [ASSISTANT]

        Args:
            user_query: The current user message.
            context:    Retrieved memory context string.
            history:    List of {'role': str, 'content': str} dicts.

        Returns:
            Full prompt string (truncated to fit n_ctx).
        """
        parts: List[str] = []

        # System instructions
        parts.append(f"[SYSTEM]\n{SYSTEM_PROMPT}\n")

        # Retrieved memory context
        if context.strip():
            parts.append(f"[RELEVANT MEMORY CONTEXT]\n{context}\n")

        # Recent conversation history (most recent kept, older dropped if tight)
        if history:
            hist_parts: List[str] = []
            for turn in history[-10:]:   # cap at last 10 turns
                role    = turn["role"].upper()
                content = truncate_to_token_budget(turn["content"], max_tokens=200)
                hist_parts.append(f"[{role}]\n{content}")
            parts.append("\n".join(hist_parts))

        # Current user query
        parts.append(f"[USER]\n{user_query}")
        parts.append("[ASSISTANT]")

        full_prompt = "\n\n".join(parts)

        # Hard truncate to stay within context window
        # (rough: 4 chars ≈ 1 token; keep ~80 % of window for prompt)
        return truncate_to_token_budget(full_prompt, int(self.n_ctx * 0.80))

    # ─────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────
    def chat(
        self,
        user_query: str,
        context: str                  = "",
        history: Optional[List[dict]] = None,
    ) -> str:
        """
        Generate a response to the user's query.

        Args:
            user_query: The user's message.
            context:    Pre-formatted memory context string (from Retriever).
            history:    Conversation history list.

        Returns:
            Generated response string.
        """
        prompt = self._build_prompt(user_query, context, history)

        if self._fallback or self._llm is None:
            return self._fallback_response(user_query, context)

        try:
            result = self._llm(
                prompt,
                max_tokens    = self.max_tokens,
                temperature   = self.temperature,
                stop          = ["[USER]", "[SYSTEM]", "[HUMAN]", "\n\n\n"],
                echo          = False,
            )
            answer = result["choices"][0]["text"].strip()
            return answer if answer else "I couldn't generate a response. Please try again."

        except Exception as exc:
            cprint(f"⚠  LLM inference error: {exc}", Colors.RED)
            return f"(LLM error: {exc})"

    def stream_chat(
        self,
        user_query: str,
        context: str                  = "",
        history: Optional[List[dict]] = None,
    ) -> Iterator[str]:
        """
        Streaming variant — yields response tokens one-by-one.

        Useful for CLI with live token display.

        Yields:
            Individual token strings.
        """
        prompt = self._build_prompt(user_query, context, history)

        if self._fallback or self._llm is None:
            response = self._fallback_response(user_query, context)
            for word in response.split(" "):
                yield word + " "
            return

        try:
            for chunk in self._llm(
                prompt,
                max_tokens  = self.max_tokens,
                temperature = self.temperature,
                stop        = ["[USER]", "[SYSTEM]", "\n\n\n"],
                echo        = False,
                stream      = True,
            ):
                token = chunk["choices"][0]["text"]
                if token:
                    yield token

        except Exception as exc:
            yield f"\n(stream error: {exc})"

    # ─────────────────────────────────────────
    # Fallback mode
    # ─────────────────────────────────────────
    def _fallback_response(self, query: str, context: str) -> str:
        """
        Simple template response used when no LLM model is loaded.

        Returns a structured answer that surfaces the retrieved context
        so the system is still useful for retrieval-only workflows.
        """
        if context.strip():
            return (
                f"[FALLBACK MODE — no LLM loaded]\n\n"
                f"Query: {query}\n\n"
                f"Here is what I found in memory:\n\n{context}"
            )
        return (
            f"[FALLBACK MODE — no LLM loaded]\n\n"
            f"Query: {query}\n\n"
            f"No relevant memories found. "
            f"Add some notes or load a .gguf model for full AI responses."
        )

    # ─────────────────────────────────────────
    # Context summarisation (for long history)
    # ─────────────────────────────────────────
    def summarise(self, text: str) -> str:
        """
        Summarise a block of text — used to compress old conversation history.

        Falls back to returning the first 300 words if no LLM is available.
        """
        if self._fallback or self._llm is None:
            words = text.split()
            return " ".join(words[:300]) + (" …[truncated]" if len(words) > 300 else "")

        prompt = (
            f"Summarise the following conversation history in 3-5 concise bullet points. "
            f"Capture key topics, decisions, and facts mentioned.\n\n{text}\n\nSummary:"
        )
        try:
            result = self._llm(
                prompt,
                max_tokens  = 256,
                temperature = 0.3,
                echo        = False,
            )
            return result["choices"][0]["text"].strip()
        except Exception:
            return " ".join(text.split()[:300])

    @property
    def is_real(self) -> bool:
        """True if a real LLM model is loaded (not fallback mode)."""
        return not self._fallback
