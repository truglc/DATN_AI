from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
DB = "database.db"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ===== INIT FOLDER =====
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ===== DB INIT =====
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            created_at TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            message TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()

init_db()

# ===== SAVE =====
def save_video(file):
    filename = file.filename

    # tránh trùng tên file
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{filename}"

    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("INSERT INTO videos (filename, created_at) VALUES (?, ?)",
              (filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()


def save_alert(video_id):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("INSERT INTO alerts (video_id, message, created_at) VALUES (?, ?, ?)",
              (video_id, "Violence detected",
               datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# ===== GET DATA =====
def get_videos():
    conn = sqlite3.connect(DB)
    data = conn.execute("SELECT * FROM videos ORDER BY id DESC").fetchall()
    conn.close()
    return data


def get_alerts():
    conn = sqlite3.connect(DB)
    data = conn.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 20").fetchall()
    conn.close()
    return data


def get_chart_hour():
    conn = sqlite3.connect(DB)
    data = conn.execute("""
        SELECT substr(created_at, 12, 2) as hour, COUNT(*)
        FROM alerts
        GROUP BY hour
        ORDER BY hour
    """).fetchall()
    conn.close()

    labels = [d[0] for d in data]
    values = [d[1] for d in data]

    return labels, values


def get_chart_day():
    conn = sqlite3.connect(DB)
    data = conn.execute("""
        SELECT substr(created_at, 1, 10) as day, COUNT(*)
        FROM alerts
        GROUP BY day
        ORDER BY day DESC
        LIMIT 7
    """).fetchall()
    conn.close()

    labels = [d[0] for d in data][::-1]
    values = [d[1] for d in data][::-1]

    return labels, values

# ===== ROUTES =====
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist("videos")
        for f in files:
            if f and f.filename != "":
                save_video(f)

        return redirect(url_for("index"))

    labels_hour, values_hour = get_chart_hour()
    labels_day, values_day = get_chart_day()

    return render_template("index.html",
                           videos=get_videos(),
                           alerts=get_alerts(),
                           labels=labels_hour,
                           values=values_hour,
                           labels_day=labels_day,
                           values_day=values_day)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/fake_alert/<int:video_id>")
def fake_alert(video_id):
    save_alert(video_id)
    return "OK"


# ===== EXTRA ROUTES (CHO MENU) =====
@app.route("/videos")
def videos_page():
    return render_template("videos.html", videos=get_videos())


@app.route("/alerts")
def alerts_page():
    return render_template("alerts.html", alerts=get_alerts())


@app.route("/history")
def history_page():
    return render_template("history.html", alerts=get_alerts())


@app.route("/settings")
def settings_page():
    return render_template("settings.html")


# ===== RUN =====
if __name__ == "__main__":
    app.run(debug=True)