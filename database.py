import sqlite3
import numpy as np
import io
import datetime
import os

DB_FILE = "attendance.db"

# Helpers to convert numpy arrays to SQLite BLOB and back
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

# Register SQLite adapters
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("ARRAY", convert_array)

def get_connection():
    # detect types to convert back arrays
    return sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)

def initialize_database():
    """Initializes the SQLite database with Users and Attendance tables."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Table: Users
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding ARRAY NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table: Attendance
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT DEFAULT 'Present',
            FOREIGN KEY(user_id) REFERENCES Users(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def add_user(name, embedding):
    """Adds a new user and their face embedding to the DB."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO Users (name, embedding) VALUES (?, ?)", (name, embedding))
    conn.commit()
    conn.close()

def get_all_users():
    """Returns a list of tuples (id, name, embedding) for all users."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, embedding FROM Users")
    users = cursor.fetchall()
    conn.close()
    return users

def mark_attendance(user_id):
    """
    Marks attendance for a user if they haven't been marked today.
    Returns True if successfully marked, False if already marked.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    # Check if already marked today
    cursor.execute("SELECT id FROM Attendance WHERE user_id = ? AND date = ?", (user_id, date_str))
    record = cursor.fetchone()
    
    if record:
        conn.close()
        return False  # Already marked
    
    # Insert new record
    cursor.execute("INSERT INTO Attendance (user_id, date, time) VALUES (?, ?, ?)", (user_id, date_str, time_str))
    conn.commit()
    conn.close()
    return True

def get_attendance_logs(date=None):
    """Fetch attendance logs. Optionally filter by date."""
    conn = get_connection()
    cursor = conn.cursor()
    
    query = '''
        SELECT a.id, u.name, a.date, a.time, a.status 
        FROM Attendance a
        JOIN Users u ON a.user_id = u.id
    '''
    params = ()
    
    if date:
        query += " WHERE a.date = ?"
        params = (date,)
        
    query += " ORDER BY a.date DESC, a.time DESC"
    
    cursor.execute(query, params)
    logs = cursor.fetchall()
    conn.close()
    return logs

# Initialize DB on import
if not os.path.exists(DB_FILE):
    initialize_database()
