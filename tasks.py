"""
tasks.py — Mini Jarvis Task & Reminder System
===============================================
Allows users to create, list, complete, and delete tasks using natural language.
Runs a lightweight background thread that polls for due reminders every 30 seconds.

Features:
  - Natural-language due time parsing ("tomorrow at 5 PM")
  - SQLite-backed persistence (tasks survive restarts)
  - Background scheduler thread
  - Desktop notification via plyer (optional), falls back to terminal bell + print
  - Task CRUD: create, list, complete, delete
"""

import sqlite3
import threading
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

from utils import extract_due_time, now_str, parse_datetime_str, cprint, Colors

# Poll interval in seconds — check for due tasks every 30 s
POLL_INTERVAL = 30


class TaskManager:
    """
    Task storage, scheduling, and notification engine.

    Usage:
        tm = TaskManager()
        tm.start_scheduler()    # start background thread

        task_id = tm.create_task("Submit report tomorrow at 10 AM")
        tm.list_tasks()
        tm.complete_task(task_id)

        tm.stop_scheduler()
    """

    def __init__(
        self,
        db_path: str                              = "minijarvis.db",
        notify_callback: Optional[Callable]       = None,
    ):
        """
        Args:
            db_path:          Path to the shared SQLite database.
            notify_callback:  Optional function called when a task is due.
                              Signature: callback(task_dict) → None.
                              If None, uses built-in desktop/terminal notification.
        """
        self.db_path         = db_path
        self.notify_callback = notify_callback or self._default_notify
        self._stop_event     = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._ensure_table()

    # ─────────────────────────────────────────
    # Database bootstrap
    # ─────────────────────────────────────────
    def _ensure_table(self) -> None:
        """Create tasks table if it doesn't already exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                content    TEXT    NOT NULL,
                due_time   TEXT,                 -- 'YYYY-MM-DD HH:MM:SS' or NULL
                created_at TEXT    NOT NULL,
                completed  INTEGER DEFAULT 0,    -- 0 = pending, 1 = done
                reminded   INTEGER DEFAULT 0     -- 0 = not yet notified
            )
        """)
        conn.commit()
        conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ─────────────────────────────────────────
    # Task CRUD
    # ─────────────────────────────────────────
    def create_task(self, text: str, due_time: Optional[datetime] = None) -> int:
        """
        Create a new task, auto-extracting due time from the text if not supplied.

        Args:
            text:     Full task description (natural language).
            due_time: Explicit due datetime. If None, extracted from text.

        Returns:
            Integer row ID of the new task.
        """
        if due_time is None:
            due_time = extract_due_time(text)

        due_str    = due_time.strftime("%Y-%m-%d %H:%M:%S") if due_time else None
        created_at = now_str()

        conn = self._get_conn()
        cur  = conn.execute(
            "INSERT INTO tasks (content, due_time, created_at) VALUES (?, ?, ?)",
            (text, due_str, created_at),
        )
        task_id = cur.lastrowid
        conn.commit()
        conn.close()

        # Friendly confirmation
        if due_time:
            cprint(f"✓  Task #{task_id} created — due: {due_str}", Colors.GREEN)
        else:
            cprint(f"✓  Task #{task_id} created (no due time detected)", Colors.GREEN)

        return task_id

    def list_tasks(
        self,
        include_completed: bool = False,
        limit: int              = 20,
    ) -> List[Dict]:
        """
        Retrieve tasks from SQLite.

        Args:
            include_completed: If False, only returns pending tasks.
            limit:             Maximum number of tasks to return.

        Returns:
            List of task dicts, ordered by due_time ASC (NULLs last).
        """
        conn  = self._get_conn()
        where = "" if include_completed else "WHERE completed = 0"
        rows  = conn.execute(
            f"""
            SELECT * FROM tasks
            {where}
            ORDER BY
                CASE WHEN due_time IS NULL THEN 1 ELSE 0 END,
                due_time ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def complete_task(self, task_id: int) -> bool:
        """
        Mark a task as completed.

        Args:
            task_id: The task's integer ID.

        Returns:
            True if the task was found and updated.
        """
        conn = self._get_conn()
        cur  = conn.execute(
            "UPDATE tasks SET completed = 1 WHERE id = ?", (task_id,)
        )
        updated = cur.rowcount > 0
        conn.commit()
        conn.close()

        if updated:
            cprint(f"✓  Task #{task_id} marked complete.", Colors.GREEN)
        else:
            cprint(f"⚠  Task #{task_id} not found.", Colors.YELLOW)

        return updated

    def delete_task(self, task_id: int) -> bool:
        """
        Permanently delete a task.

        Args:
            task_id: The task's integer ID.

        Returns:
            True if deleted.
        """
        conn = self._get_conn()
        cur  = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        deleted = cur.rowcount > 0
        conn.commit()
        conn.close()

        if deleted:
            cprint(f"✓  Task #{task_id} deleted.", Colors.GREEN)
        else:
            cprint(f"⚠  Task #{task_id} not found.", Colors.YELLOW)

        return deleted

    def get_task(self, task_id: int) -> Optional[Dict]:
        """Fetch a single task by ID."""
        conn = self._get_conn()
        row  = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        conn.close()
        return dict(row) if row else None

    # ─────────────────────────────────────────
    # Overdue & upcoming helpers
    # ─────────────────────────────────────────
    def get_due_tasks(self) -> List[Dict]:
        """
        Return all pending tasks whose due_time is in the past and
        haven't been notified yet.

        Returns:
            List of task dicts.
        """
        now  = now_str()
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM tasks
             WHERE completed = 0
               AND reminded  = 0
               AND due_time IS NOT NULL
               AND due_time <= ?
            """,
            (now,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_upcoming_tasks(self, hours_ahead: int = 24) -> List[Dict]:
        """
        Return pending tasks due within the next N hours.

        Args:
            hours_ahead: Lookahead window in hours.

        Returns:
            List of task dicts sorted by due_time.
        """
        from datetime import timedelta
        now_dt  = datetime.now()
        until   = (now_dt + timedelta(hours=hours_ahead)).strftime("%Y-%m-%d %H:%M:%S")
        now_str_ = now_dt.strftime("%Y-%m-%d %H:%M:%S")

        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM tasks
             WHERE completed = 0
               AND due_time IS NOT NULL
               AND due_time BETWEEN ? AND ?
             ORDER BY due_time ASC
            """,
            (now_str_, until),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ─────────────────────────────────────────
    # Notification
    # ─────────────────────────────────────────
    def _default_notify(self, task: Dict) -> None:
        """
        Built-in notification handler.

        Tries plyer (cross-platform desktop notifications) first.
        Falls back to terminal output with a bell character.
        """
        title   = "⏰ Mini Jarvis Reminder"
        message = task["content"]

        # ── Try desktop notification ───────────────────────────────
        try:
            from plyer import notification  # type: ignore
            notification.notify(
                title   = title,
                message = message,
                app_name= "Mini Jarvis",
                timeout = 10,
            )
        except Exception:
            pass  # plyer not installed or unsupported

        # ── Always print to terminal too ───────────────────────────
        print("\a", end="", flush=True)  # terminal bell
        cprint(f"\n{'─'*50}", Colors.YELLOW, bold=True)
        cprint(f"  {title}", Colors.YELLOW, bold=True)
        cprint(f"  Task #{task['id']}: {message}", Colors.YELLOW)
        cprint(f"  Due: {task['due_time']}", Colors.YELLOW)
        cprint(f"{'─'*50}\n", Colors.YELLOW, bold=True)

    def _mark_reminded(self, task_id: int) -> None:
        """Mark a task as having been notified to avoid duplicate alerts."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE tasks SET reminded = 1 WHERE id = ?", (task_id,))
        conn.commit()
        conn.close()

    # ─────────────────────────────────────────
    # Background scheduler
    # ─────────────────────────────────────────
    def _scheduler_loop(self) -> None:
        """
        Background daemon thread that polls for due tasks every POLL_INTERVAL seconds.
        Fires the notification callback for each newly-due task.
        """
        while not self._stop_event.is_set():
            try:
                due_tasks = self.get_due_tasks()
                for task in due_tasks:
                    self.notify_callback(task)
                    self._mark_reminded(task["id"])
            except Exception as exc:
                # Never crash the scheduler thread
                cprint(f"⚠  Scheduler error: {exc}", Colors.RED)

            self._stop_event.wait(POLL_INTERVAL)

    def start_scheduler(self) -> None:
        """Start the background reminder polling thread."""
        if self._thread and self._thread.is_alive():
            cprint("⚠  Scheduler already running.", Colors.YELLOW)
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target  = self._scheduler_loop,
            name    = "MiniJarvis-Scheduler",
            daemon  = True,   # auto-killed when main thread exits
        )
        self._thread.start()
        cprint("✓  Task scheduler started (polling every 30 s)", Colors.GREEN)

    def stop_scheduler(self) -> None:
        """Signal the background scheduler thread to stop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        cprint("✓  Task scheduler stopped.", Colors.GREEN)

    # ─────────────────────────────────────────
    # Display helper
    # ─────────────────────────────────────────
    def print_tasks(self, include_completed: bool = False) -> None:
        """Pretty-print pending (or all) tasks to the terminal."""
        tasks = self.list_tasks(include_completed=include_completed)

        if not tasks:
            cprint(
                "  No pending tasks." if not include_completed else "  No tasks found.",
                Colors.YELLOW,
            )
            return

        label = "All tasks" if include_completed else "Pending tasks"
        cprint(f"\n  ── {label} ({len(tasks)}) ──\n", Colors.CYAN, bold=True)

        for t in tasks:
            status = "✓" if t["completed"] else "○"
            due    = t["due_time"] or "no due time"
            color  = Colors.GREEN if t["completed"] else Colors.BLUE
            cprint(f"  {status}  #{t['id']:3d}  {t['content'][:60]}", color)
            print(f"           Due: {due}  |  Created: {t['created_at']}")
        print()
