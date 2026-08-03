from flask import Flask, request, jsonify, render_template, send_from_directory
import sqlite3

app = Flask(__name__)

# =====================================
# DATABASE SETUP
# =====================================

def init_db():

    conn = sqlite3.connect("traffic.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS fines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate TEXT,
        violation TEXT,
        amount INTEGER,
        image_path TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'unpaid'
    )
    """)

    conn.commit()
    conn.close()

init_db()

# =====================================
# HOME PAGE
# =====================================

@app.route("/")
def home():
    return render_template("frontend.html")

# =====================================
# SERVE EVIDENCE IMAGES
# =====================================

@app.route('/evidence/<path:filename>')
def evidence(filename):

    import os

    evidence_folder = os.path.join(
        app.root_path,
        "evidence"
    )

    return send_from_directory(
        evidence_folder,
        filename
    )

# =====================================
# SAVE DETECTED VIOLATION
# =====================================

@app.route("/detect", methods=["POST"])
def detect():

    data = request.json

    plate = data["plate"]
    violation = data["violation"]
    image_path = data["image_path"]

    fine_amounts = {
        "no_helmet": 500,
        "triple_riding": 1000
    }

    amount = fine_amounts.get(
        violation,
        300
    )

    conn = sqlite3.connect("traffic.db")
    c = conn.cursor()

    c.execute("""
    INSERT INTO fines
    (
        plate,
        violation,
        amount,
        image_path
    )
    VALUES (?, ?, ?, ?)
    """,
    (
        plate,
        violation,
        amount,
        image_path
    ))

    conn.commit()
    conn.close()

    return jsonify({
        "message": "Violation Recorded"
    })

# =====================================
# GET FINES BY PLATE
# =====================================

@app.route("/get_fines/<plate>")
def get_fines(plate):

    conn = sqlite3.connect("traffic.db")
    c = conn.cursor()

    c.execute("""
    SELECT
        violation,
        amount,
        image_path,
        timestamp,
        status
    FROM fines
    WHERE plate=?
    """, (plate,))

    rows = c.fetchall()

    conn.close()

    fines = []
    total = 0

    for row in rows:

        image_file = row[2].split("/")[-1]

        fines.append({
            "violation": row[0],
            "amount": row[1],
            "status": row[4],
            "time": row[3],
            "image": f"/evidence/{image_file}"
        })

        if row[4] == "unpaid":
            total += row[1]

    return jsonify({
        "fines": fines,
        "total": total
    })

# =====================================
# SHOW ALL FINES
# =====================================

@app.route("/fines")
def all_fines():

    conn = sqlite3.connect("traffic.db")
    c = conn.cursor()

    c.execute("SELECT * FROM fines")

    rows = c.fetchall()

    conn.close()

    return jsonify(rows)

# =====================================
# RUN
# =====================================

if __name__ == "__main__":
    app.run(debug=True)