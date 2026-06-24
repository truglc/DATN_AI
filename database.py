
# Database SQLite: tạo bảng + insert/update log

import sqlite3
from datetime import datetime
from config import DB_PATH


def get_conn():
    return sqlite3.connect(DB_PATH)


def ensure_column(cursor, table, column, definition):
    cursor.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cursor.fetchall()]
    if column not in cols:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            original_name TEXT,
            source TEXT,
            output_filename TEXT,
            created_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            source TEXT,
            label TEXT,
            confidence REAL,
            lstm_score REAL,
            rule_score REAL,
            frame_index INTEGER,
            person_count INTEGER,
            fps REAL,
            latency_ms REAL,
            snapshot TEXT,
            created_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            source TEXT,
            frame_index INTEGER,
            fps REAL,
            latency_ms REAL,
            person_count INTEGER,
            created_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            source TEXT,
            frame_index INTEGER,
            label TEXT,
            fusion_score REAL,
            lstm_score REAL,
            rule_score REAL,
            iou_score REAL,
            interaction_score REAL,
            motion_score REAL,
            fall_score REAL,
            running_score REAL,
            person_count INTEGER,
            fps REAL,
            latency_ms REAL,
            created_at TEXT
        )
    """)
    ensure_column(c, "videos", "output_filename", "TEXT")
    conn.commit()
    conn.close()


def insert_video(filename, original_name, source):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO videos(filename, original_name, source, created_at) VALUES (?, ?, ?, ?)",
        (filename, original_name, source, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    video_id = c.lastrowid
    conn.close()
    return video_id


def update_video_output(video_id, output_filename):
    if video_id is None:
        return
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE videos SET output_filename=? WHERE id=?", (output_filename, video_id))
    conn.commit()
    conn.close()


def insert_alert(video_id, source, label, confidence, lstm_score, rule_score,
                 frame_index, person_count, fps, latency_ms, snapshot):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO alerts(video_id, source, label, confidence, lstm_score, rule_score,
                           frame_index, person_count, fps, latency_ms, snapshot, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        video_id, source, label, float(confidence), float(lstm_score), float(rule_score),
        int(frame_index), int(person_count), float(fps), float(latency_ms), snapshot,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()


def insert_performance(video_id, source, frame_index, fps, latency_ms, person_count):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO performance(video_id, source, frame_index, fps, latency_ms, person_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (video_id, source, int(frame_index), float(fps), float(latency_ms), int(person_count),
          datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()


def insert_prediction_log(video_id, source, frame_index, label, fusion_score, lstm_score, rule_score,
                          iou_score, interaction_score, motion_score, fall_score, running_score,
                          person_count, fps, latency_ms):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO prediction_logs(video_id, source, frame_index, label, fusion_score, lstm_score, rule_score,
                                    iou_score, interaction_score, motion_score, fall_score, running_score,
                                    person_count, fps, latency_ms, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (video_id, source, int(frame_index), label, float(fusion_score), float(lstm_score), float(rule_score),
          float(iou_score), float(interaction_score), float(motion_score), float(fall_score), float(running_score),
          int(person_count), float(fps), float(latency_ms), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

