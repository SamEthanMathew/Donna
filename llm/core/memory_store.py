import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

@dataclass
class MemoryItem:
    id: int
    key: str
    value: str
    category: str
    confidence: float
    created_at: str
    updated_at: str
    last_used_at: Optional[str]

class MemoryStore:
    def __init__(self, db_path: Path):
        self.db_path = str(db_path)
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL UNIQUE,
                value TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.7,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_used_at TEXT
            )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_last_used ON memories(last_used_at)")
            conn.commit()

    def upsert_memory(self, key: str, value: str, category: str, confidence: float = 0.75) -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute("""
            INSERT INTO memories (key, value, category, confidence, created_at, updated_at, last_used_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
              value=excluded.value,
              category=excluded.category,
              confidence=excluded.confidence,
              updated_at=excluded.updated_at
            """, (key, value, category, confidence, now, now, None))
            conn.commit()

    def list_memories(self, limit: int = 50) -> List[MemoryItem]:
        with self._conn() as conn:
            rows = conn.execute("""
            SELECT id, key, value, category, confidence, created_at, updated_at, last_used_at
            FROM memories
            ORDER BY updated_at DESC
            LIMIT ?
            """, (limit,)).fetchall()
        return [MemoryItem(*row) for row in rows]

    def get_memory(self, key: str) -> Optional[MemoryItem]:
        with self._conn() as conn:
            row = conn.execute("""
            SELECT id, key, value, category, confidence, created_at, updated_at, last_used_at
            FROM memories
            WHERE key=?
            """, (key,)).fetchone()
        return MemoryItem(*row) if row else None

    def search_memories(self, query: str, limit: int = 10) -> List[MemoryItem]:
        q = f"%{query.lower()}%"
        with self._conn() as conn:
            rows = conn.execute("""
            SELECT id, key, value, category, confidence, created_at, updated_at, last_used_at
            FROM memories
            WHERE lower(key) LIKE ? OR lower(value) LIKE ?
            ORDER BY updated_at DESC
            LIMIT ?
            """, (q, q, limit)).fetchall()
        return [MemoryItem(*row) for row in rows]

    def mark_used(self, memory_ids: List[int]) -> None:
        if not memory_ids:
            return
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.executemany(
                "UPDATE memories SET last_used_at=? WHERE id=?",
                [(now, mid) for mid in memory_ids]
            )
            conn.commit()


