"""Consent-based audit storage in SQLite (report sections 5.7 and 6).

Nothing is written unless the caller chooses to save. Images need separate consent. Records
older than the retention period are purged whenever the store is opened.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import DEFAULT_CONFIG, Config
from .evidence import AuditRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_records (
    record_id  TEXT PRIMARY KEY,
    timestamp  TEXT NOT NULL,
    response   TEXT NOT NULL,
    consistency REAL,
    image_path TEXT,
    record     TEXT NOT NULL
)
"""


class AuditStore:
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.db_path = Path(config.audit_db)
        self.images_dir = Path(config.audit_images_dir)
        self.retention_days = config.retention_days
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)
        self.purge_expired()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def save(self, record: AuditRecord, image_bytes: bytes | None = None) -> None:
        """Insert or update a record. image_bytes is stored only if the user agreed to it."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT image_path FROM audit_records WHERE record_id = ?", (record.record_id,)
            ).fetchone()
            image_path = row[0] if row else None
            if image_bytes is not None:
                self.images_dir.mkdir(parents=True, exist_ok=True)
                path = self.images_dir / f"{record.record_id}.img"
                path.write_bytes(image_bytes)
                image_path = str(path)
            conn.execute(
                "INSERT OR REPLACE INTO audit_records VALUES (?, ?, ?, ?, ?, ?)",
                (record.record_id, record.timestamp, record.response, record.consistency,
                 image_path, json.dumps(record.to_dict())),
            )

    def list_records(self) -> list[dict]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT record_id, timestamp, response, consistency, image_path "
                "FROM audit_records ORDER BY timestamp DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get(self, record_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT record FROM audit_records WHERE record_id = ?", (record_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def delete(self, record_id: str) -> None:
        self._delete_where("record_id = ?", (record_id,))

    def delete_all(self) -> None:
        self._delete_where("1 = 1", ())

    def purge_expired(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        self._delete_where("timestamp < ?", (cutoff.isoformat(timespec="seconds"),))

    def _delete_where(self, where: str, params: tuple) -> None:
        with self._connect() as conn:
            paths = conn.execute(
                f"SELECT image_path FROM audit_records WHERE {where}", params
            ).fetchall()
            conn.execute(f"DELETE FROM audit_records WHERE {where}", params)
        for (path,) in paths:
            if path:
                Path(path).unlink(missing_ok=True)
