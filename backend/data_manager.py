"""
ForgeSight — Production Data Platform & SQLite Storage Engine
Tracks inspection events, human reviewer verification loop,
dataset versioning, and edge station telemetry.
"""

import sqlite3
import json
import time
import os
from typing import Dict, Any, List, Optional


class IndustrialDataManager:
    def __init__(self, db_path: str = "forgesight_data.db"):
        self.db_path = db_path
        self._init_db()
        self._seed_initial_stations_and_data()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Inspection Events
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id TEXT PRIMARY KEY,
                part_id TEXT NOT NULL,
                station_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                is_defect INTEGER NOT NULL,
                anomaly_score REAL NOT NULL,
                defect_type TEXT NOT NULL,
                review_status TEXT DEFAULT 'PENDING',
                model_version TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                metadata_json TEXT
            );
            """)

            # 2. Human Review Triage
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS human_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                decision TEXT NOT NULL,
                notes TEXT,
                submitted_at REAL NOT NULL,
                FOREIGN KEY (inspection_id) REFERENCES inspections (id)
            );
            """)

            # 3. Versioned Datasets for Continual Retraining
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_versions (
                version_tag TEXT PRIMARY KEY,
                total_samples INTEGER NOT NULL,
                golden_normals INTEGER NOT NULL,
                verified_defects INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                description TEXT
            );
            """)

            # 4. Factory Stations
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS stations (
                station_id TEXT PRIMARY KEY,
                station_name TEXT NOT NULL,
                location TEXT NOT NULL,
                camera_type TEXT NOT NULL,
                status TEXT NOT NULL,
                target_ppm INTEGER NOT NULL,
                last_heartbeat REAL NOT NULL
            );
            """)

            conn.commit()

    def _seed_initial_stations_and_data(self):
        """Seed default factory line stations and baseline datasets."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Stations
            stations = [
                ("STN-01-CAST", "Station 1 — Raw Castings Inspection", "Line A / Bay 1", "Basler ace 2 U (GigE)", "ONLINE", 120),
                ("STN-02-MILL", "Station 2 — CNC Machined Surfaces", "Line A / Bay 3", "Sony Pregius 5MP USB3", "ONLINE", 120),
                ("STN-03-BORE", "Station 3 — Bearing Bore & Chamfer", "Line A / Bay 5", "Flir Blackfly S (GigE)", "ONLINE", 120),
                ("STN-04-ASSY", "Station 4 — Final Packout Verification", "Line A / Bay 8", "Hikrobot Vision Cam", "ONLINE", 100),
            ]
            for s in stations:
                cursor.execute("""
                INSERT OR REPLACE INTO stations (station_id, station_name, location, camera_type, status, target_ppm, last_heartbeat)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (*s, time.time()))

            # Datasets
            cursor.execute("""
            INSERT OR REPLACE INTO dataset_versions (version_tag, total_samples, golden_normals, verified_defects, status, created_at, description)
            VALUES 
            ('mvtec-metal-parts-v1.0', 1200, 1200, 0, 'ARCHIVED', 1723450000, 'Baseline unsupervised normal-only golden set'),
            ('mvtec-metal-parts-v2.1', 2850, 2780, 70, 'ACTIVE', 1727500000, 'Expanded golden baseline + 70 initial calibrated defects'),
            ('mvtec-metal-parts-v2.4-reviewed', 3420, 3210, 210, 'STAGED', 1728700000, 'Includes 140 triage-approved edge cases from manufacturing pilot');
            """)

            conn.commit()

    def save_inspection(self, record: Dict[str, Any]) -> str:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO inspections (id, part_id, station_id, timestamp, is_defect, anomaly_score, defect_type, review_status, model_version, latency_ms, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record["id"],
                record["part_id"],
                record["station_id"],
                record["timestamp"],
                1 if record["is_defect"] else 0,
                record["anomaly_score"],
                record["defect_type"],
                record.get("review_status", "PENDING" if record["is_defect"] else "AUTO_PASSED"),
                record["model_version"],
                record["latency_ms"],
                json.dumps(record.get("metadata", {}))
            ))
            conn.commit()
        return record["id"]

    def submit_review(self, inspection_id: str, reviewer: str, decision: str, notes: str = "") -> Dict[str, Any]:
        """
        Record inspector decision:
        'CONFIRM_DEFECT', 'FALSE_ALARM', or 'UNCERTAIN'
        Updates review_status and feeds continual learning pipeline.
        """
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO human_reviews (inspection_id, reviewer, decision, notes, submitted_at)
            VALUES (?, ?, ?, ?, ?)
            """, (inspection_id, reviewer, decision, notes, now))

            new_status = f"VERIFIED_{decision}"
            cursor.execute("""
            UPDATE inspections SET review_status = ? WHERE id = ?
            """, (new_status, inspection_id))

            conn.commit()

        return {
            "inspection_id": inspection_id,
            "decision": decision,
            "review_status": new_status,
            "reviewer": reviewer,
            "submitted_at": now
        }

    def get_recent_inspections(self, limit: int = 30) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM inspections ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_triage_queue(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Fetch items needing expert human review (flagged defects or uncertain borderline scores)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM inspections 
            WHERE review_status = 'PENDING' OR (anomaly_score >= 0.40 AND anomaly_score <= 0.60)
            ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
            return [dict(r) for r in cursor.fetchall()]

    def get_stations(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM stations")
            return [dict(r) for r in cursor.fetchall()]

    def get_dataset_versions(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM dataset_versions ORDER BY created_at DESC")
            return [dict(r) for r in cursor.fetchall()]
