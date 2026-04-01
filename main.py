"""
main.py — Mini Jarvis Entry Point
===================================
Wires together all modules and exposes a rich CLI.

Commands:
  chat     — Start an interactive conversation with Mini Jarvis
  add      — Add a note or task directly
  search   — Semantic search over stored memories
  list     — List all memories or tasks
  tasks    — Task management subcommands
  stats    — Show memory + system statistics
  delete   — Delete a memory by ID
  history  — View conversation history

Run:
  python main.py                  # interactive chat (default)
  python main.py --help
  python main.py chat
  python main.py add "Remember to read about FAISS" --type note
  python main.py search "python notes"
  python main.py tasks list
  python main.py stats
"""

import os
import sys
import argparse
import textwrap
from typing import Optional

from memory       import MemoryStore
from retrieval    import Retriever
from llm          import LocalLLM
from conversation import ConversationManager
from tasks        import TaskManager
from utils        import cprint, Colors, now_str

# ─────────────────────────────────────────────
# Configuration — edit these before running
# ─────────────────────────────────────────────
MODEL_PATH   = os.environ.get("JARVIS_MODEL", "C:\coding\models\Phi-3-mini-4k-instruct-q4.gguf")
DB_PATH      = os.environ.get("JARVIS_DB",    "minijarvis.db")
N_CTX        = int(os.environ.get("JARVIS_CTX",     "4096"))
N_THREADS    = int(os.environ.get("JARVIS_THREADS",  "4"))
N_GPU_LAYERS = int(os.environ.get("JARVIS_GPU",      "0"))

BANNER = r"""
  __  __ _       _       _                  _     
 |  \/  (_)_ __ (_)     | | __ _ _ ____   _(_)___ 
 | |\/| | | '_ \| |  _  | |/ _` | '__\ \ / / / __|
 | |  | | | | | | | | |_| | (_| | |   \ V /| \__ \
 |_|  |_|_|_| |_|_|  \___/ \__,_|_|    \_/ |_|___/
                                                   
  Your local-first, privacy-first AI assistant
  Type  :help  for commands,  :quit  to exit.
"""


# ─────────────────────────────────────────────
# Mini Jarvis application class
# ─────────────────────────────────────────────
class MiniJarvis:
    """
    Central application object that owns all module instances and
    orchestrates the query pipeline.

    Pipeline (per user message):
      1. Store the user message as a 'conversation' memory entry
      2. Embed the query → FAISS search → retrieve top-k memories
      3. Build context string from retrieved memories
      4. Inject context + conversation history into LocalLLM
      5. Generate response
      6. Store response as 'conversation' memory entry
    """

    def __init__(self, model_path: str = MODEL_PATH):
        cprint("\n" + "─" * 56, Colors.CYAN)
        cprint("  Initialising Mini Jarvis…", Colors.CYAN, bold=True)
        cprint("─" * 56, Colors.CYAN)

        self.store   = MemoryStore(db_path=DB_PATH)
        self.retriever = Retriever(self.store)
        self.llm     = LocalLLM(
            model_path   = model_path,
            n_ctx        = N_CTX,
            n_threads    = N_THREADS,
            n_gpu_layers = N_GPU_LAYERS,
        )
        self.conv    = ConversationManager(db_path=DB_PATH)
        self.tasks   = TaskManager(db_path=DB_PATH)

        cprint("─" * 56 + "\n", Colors.CYAN)

    # ─────────────────────────────────────────
    # Core: ask
    # ─────────────────────────────────────────
    def ask(self, query: str, stream: bool = True) -> str:
        """
        Full RAG pipeline: retrieve → prompt → respond.

        Args:
            query:  The user's natural-language query.
            stream: If True and LLM supports it, stream tokens to stdout.

        Returns:
            The assistant's response string.
        """
        # 1. Persist user turn
        self.conv.add_turn("user", query)
        self.store.add_memory(query, mem_type="conversation")

        # 2. Retrieve relevant memories
        context = self.retriever.retrieve_context(query, top_k=5, max_tokens=1200)

        # 3. Get conversation history for LLM
        history = self.conv.get_history_for_prompt(max_turns=10)

        # 4. Generate response
        if stream and self.llm.is_real:
            cprint("\nJarvis: ", Colors.GREEN, bold=True)
            print("", end="", flush=True)
            response_parts = []
            for token in self.llm.stream_chat(query, context, history):
                print(token, end="", flush=True)
                response_parts.append(token)
            print()  # newline after streaming
            response = "".join(response_parts).strip()
        else:
            response = self.llm.chat(query, context, history)

        # 5. Persist assistant turn
        self.conv.add_turn("assistant", response)
        self.store.add_memory(response, mem_type="conversation")

        return response

    # ─────────────────────────────────────────
    # Memory management shortcuts
    # ─────────────────────────────────────────
    def add_note(self, text: str, tags: str = "") -> None:
        """Store a freeform note."""
        ids = self.store.add_memory(text, mem_type="note", tags=tags)
        cprint(f"✓  Note saved (id: {ids[0]})", Colors.GREEN)

    def add_task_note(self, text: str) -> None:
        """Store a task description as a memory AND schedule it."""
        ids     = self.store.add_memory(text, mem_type="task")
        task_id = self.tasks.create_task(text)
        cprint(f"✓  Task memory saved (id: {ids[0]}) and scheduled (task #{task_id})", Colors.GREEN)

    def search(self, query: str, top_k: int = 5, mem_type: Optional[str] = None) -> None:
        """Run a semantic search and print results."""
        results = self.retriever.search(query, top_k=top_k, mem_type=mem_type)
        self.retriever.print_results(results)

    def delete_memory(self, mem_id: str) -> None:
        """Delete a memory by ID."""
        ok = self.store.delete_memory(mem_id)
        if ok:
            cprint(f"✓  Memory '{mem_id}' deleted.", Colors.GREEN)
        else:
            cprint(f"⚠  Memory '{mem_id}' not found.", Colors.YELLOW)

    def show_stats(self) -> None:
        """Print memory and session statistics."""
        stats = self.store.stats()
        cprint("\n  ── Memory statistics ──\n", Colors.CYAN, bold=True)
        for key, val in stats.items():
            print(f"    {key:<20} {val}")

        session_count = len(self.conv.get_all_sessions())
        print(f"    {'sessions':<20} {session_count}")

        tasks_pending = len(self.tasks.list_tasks(include_completed=False))
        print(f"    {'pending tasks':<20} {tasks_pending}")
        print()

    # ─────────────────────────────────────────
    # Interactive REPL
    # ─────────────────────────────────────────
    def repl(self) -> None:
        """
        Start the interactive command-line REPL.

        Special commands begin with ':':
          :quit / :q      — exit
          :help / :h      — show commands
          :note <text>    — add a note directly
          :task <text>    — add and schedule a task
          :search <query> — semantic search
          :list [type]    — list memories
          :tasks          — show pending tasks
          :history        — show conversation history
          :stats          — show statistics
          :new            — start a new conversation session
          :clear          — clear current session history
          :delete <id>    — delete a memory
        """
        print(BANNER)
        if not self.llm.is_real:
            cprint(
                "  ⚠  Running in fallback mode (no LLM model).\n"
                "     Set JARVIS_MODEL=/path/to/model.gguf to enable full AI responses.\n",
                Colors.YELLOW,
            )

        # Start the background task scheduler
        self.tasks.start_scheduler()

        try:
            while True:
                try:
                    user_input = input(f"\n{Colors.BOLD}{Colors.BLUE}You: {Colors.RESET}").strip()
                except (KeyboardInterrupt, EOFError):
                    cprint("\n\nGoodbye! 👋", Colors.CYAN, bold=True)
                    break

                if not user_input:
                    continue

                # ── Special commands ───────────────────────────────
                if user_input.startswith(":"):
                    self._handle_command(user_input)
                    continue

                # ── Regular chat ───────────────────────────────────
                if not self.llm.is_real:
                    # In fallback mode, print the full response manually
                    response = self.ask(user_input, stream=False)
                    cprint(f"\nJarvis: {response}", Colors.GREEN, bold=True)
                else:
                    self.ask(user_input, stream=True)

        finally:
            self.tasks.stop_scheduler()

    def _handle_command(self, cmd: str) -> None:
        """
        Dispatch REPL special commands.

        Args:
            cmd: Raw input string starting with ':'.
        """
        parts = cmd.split(maxsplit=1)
        verb  = parts[0].lower()
        arg   = parts[1].strip() if len(parts) > 1 else ""

        if verb in (":quit", ":q", ":exit"):
            cprint("\nGoodbye! 👋", Colors.CYAN, bold=True)
            sys.exit(0)

        elif verb in (":help", ":h"):
            self._print_help()

        elif verb == ":note":
            if arg:
                self.add_note(arg)
            else:
                cprint("  Usage: :note <text>", Colors.YELLOW)

        elif verb == ":task":
            if arg:
                self.add_task_note(arg)
            else:
                cprint("  Usage: :task <text>", Colors.YELLOW)

        elif verb == ":search":
            if arg:
                self.search(arg)
            else:
                cprint("  Usage: :search <query>", Colors.YELLOW)

        elif verb == ":list":
            mem_type = arg if arg in ("note", "task", "conversation") else None
            mems = self.store.get_all_memories(mem_type=mem_type)
            if not mems:
                cprint("  No memories stored yet.", Colors.YELLOW)
            else:
                cprint(f"\n  ── Stored memories ({len(mems)}) ──\n", Colors.CYAN, bold=True)
                for m in mems[:30]:   # cap display at 30
                    preview = m["content"][:80] + ("…" if len(m["content"]) > 80 else "")
                    cprint(f"  [{m['type'].upper()}] {m['id']} | {m['timestamp']}", Colors.BLUE)
                    print(f"  {preview}\n")

        elif verb == ":tasks":
            self.tasks.print_tasks()

        elif verb == ":history":
            self.conv.print_history()

        elif verb == ":stats":
            self.show_stats()

        elif verb == ":new":
            self.conv.new_session()

        elif verb == ":clear":
            self.conv.clear_session()

        elif verb == ":delete":
            if arg:
                self.delete_memory(arg)
            else:
                cprint("  Usage: :delete <memory_id>", Colors.YELLOW)

        else:
            cprint(f"  Unknown command: {verb}  (type :help for options)", Colors.YELLOW)

    def _print_help(self) -> None:
        """Print the REPL command reference."""
        cprint("\n  ── Mini Jarvis Commands ──\n", Colors.CYAN, bold=True)
        commands = [
            (":note <text>",     "Save a note to memory"),
            (":task <text>",     "Create a reminder/task (parses due time automatically)"),
            (":search <query>",  "Semantic search over all memories"),
            (":list [type]",     "List memories — type: note | task | conversation"),
            (":tasks",           "Show pending tasks"),
            (":history",         "View recent conversation turns"),
            (":stats",           "Show memory and session statistics"),
            (":new",             "Start a new conversation session"),
            (":clear",           "Clear current session history"),
            (":delete <id>",     "Delete a memory by its ID"),
            (":help",            "Show this help message"),
            (":quit",            "Exit Mini Jarvis"),
        ]
        for cmd, desc in commands:
            cprint(f"  {cmd:<25}", Colors.BLUE, bold=True)
            print(f"    {desc}")
        print()


# ─────────────────────────────────────────────
# CLI argument parser
# ─────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    """Build and return the argparse CLI parser."""
    parser = argparse.ArgumentParser(
        prog        = "mini_jarvis",
        description = "Mini Jarvis — local-first personal AI assistant",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = textwrap.dedent("""
            Examples:
              python main.py                               # interactive chat
              python main.py chat                          # same
              python main.py add "My note here"            # add a note
              python main.py add "Fix bug by tomorrow 5pm" --type task
              python main.py search "python notes"
              python main.py list --type note
              python main.py tasks list
              python main.py tasks done 3
              python main.py stats
        """),
    )

    parser.add_argument(
        "--model", "-m",
        default = MODEL_PATH,
        help    = "Path to .gguf LLM model file",
    )

    subparsers = parser.add_subparsers(dest="command")

    # chat
    subparsers.add_parser("chat", help="Start interactive chat (default)")

    # add
    add_p = subparsers.add_parser("add", help="Add a note or task")
    add_p.add_argument("text", help="Content to store")
    add_p.add_argument(
        "--type", "-t",
        choices = ["note", "task"],
        default = "note",
        dest    = "mem_type",
        help    = "Memory type (default: note)",
    )
    add_p.add_argument("--tags", default="", help="Comma-separated tags")

    # search
    search_p = subparsers.add_parser("search", help="Semantic search")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument("--top", "-k", type=int, default=5, help="Top-K results")
    search_p.add_argument(
        "--type", "-t",
        choices = ["note", "task", "conversation"],
        default = None,
        dest    = "mem_type",
        help    = "Filter by memory type",
    )

    # list
    list_p = subparsers.add_parser("list", help="List stored memories")
    list_p.add_argument(
        "--type", "-t",
        choices = ["note", "task", "conversation"],
        default = None,
        dest    = "mem_type",
    )

    # stats
    subparsers.add_parser("stats", help="Show statistics")

    # tasks
    tasks_p    = subparsers.add_parser("tasks", help="Task management")
    tasks_sub  = tasks_p.add_subparsers(dest="tasks_cmd")

    tasks_sub.add_parser("list",  help="List pending tasks")
    tasks_sub.add_parser("all",   help="List all tasks including completed")

    done_p = tasks_sub.add_parser("done",   help="Mark task complete")
    done_p.add_argument("task_id", type=int)

    del_p  = tasks_sub.add_parser("delete", help="Delete a task")
    del_p.add_argument("task_id", type=int)

    return parser


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    jarvis = MiniJarvis(model_path=args.model if hasattr(args, "model") else MODEL_PATH)

    cmd = getattr(args, "command", None)

    if cmd is None or cmd == "chat":
        jarvis.repl()

    elif cmd == "add":
        if args.mem_type == "task":
            jarvis.add_task_note(args.text)
        else:
            jarvis.add_note(args.text, tags=args.tags)

    elif cmd == "search":
        jarvis.search(args.query, top_k=args.top, mem_type=args.mem_type)

    elif cmd == "list":
        mems = jarvis.store.get_all_memories(mem_type=args.mem_type)
        if not mems:
            cprint("  No memories found.", Colors.YELLOW)
        else:
            cprint(f"\n  ── {len(mems)} memories ──\n", Colors.CYAN, bold=True)
            for m in mems[:50]:
                preview = m["content"][:100] + ("…" if len(m["content"]) > 100 else "")
                cprint(f"  [{m['type'].upper()}] {m['id']} | {m['timestamp']}", Colors.BLUE)
                print(f"  {preview}\n")

    elif cmd == "stats":
        jarvis.show_stats()

    elif cmd == "tasks":
        tc = getattr(args, "tasks_cmd", None)
        if tc == "list":
            jarvis.tasks.print_tasks(include_completed=False)
        elif tc == "all":
            jarvis.tasks.print_tasks(include_completed=True)
        elif tc == "done":
            jarvis.tasks.complete_task(args.task_id)
        elif tc == "delete":
            jarvis.tasks.delete_task(args.task_id)
        else:
            jarvis.tasks.print_tasks()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
