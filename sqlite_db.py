import os
import sqlite3
import sys
from collections import defaultdict

"""
Database Schema:

1. audio_files table:
   - audio_id INTEGER PRIMARY KEY AUTOINCREMENT
   - file_path TEXT UNIQUE NOT NULL
   - filename TEXT NOT NULL

2. Each fingerprint algorithm has its own table.
2a. maxima_pairing_fingerprints table:
    - hash_hex TEXT NOT NULL
    - anchor_time INTEGER NOT NULL
    - audio_id INTEGER NOT NULL
    - FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id)
    
Indices:
- On maxima_pairing_fingerprints(hash_hex) for faster lookup during matching
- On audio_files(file_path) for checking duplicates
"""

class SQLiteDB:
    def __init__(self, db_path='fingerprints.db', clear_db=False):
        """Initializes the connection and creates tables if they don't exist."""
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        if clear_db:
            self._clear_db()
        self._create_tables()

    def _create_tables(self):
        """Creates the necessary tables and indices if they don't already exist."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS audio_files (
                audio_id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL
            )
        ''')

        # Create a separate fingerprints table for MaximaPairingAlgorithm.
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS maxima_pairing_fingerprints (
                hash_hex TEXT NOT NULL,
                anchor_time INTEGER NOT NULL,
                audio_id INTEGER NOT NULL,
                FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id) ON DELETE CASCADE
            )
        ''')
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_maxima_pairing_hash
            ON maxima_pairing_fingerprints (hash_hex)
        ''')
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_audio_files_path
            ON audio_files (file_path)
        ''')
        self.conn.commit()

    def _clear_db(self):
        """Removes all data from the database by dropping tables."""
        print(f"Clearing database '{self.db_path}'...")
        self.cursor.execute('DROP TABLE IF EXISTS maxima_pairing_fingerprints')
        self.cursor.execute('DROP TABLE IF EXISTS audio_files')
        self.conn.commit()
        self._create_tables() 
        print("Database cleared and schema recreated.")
        self.cursor.execute('SELECT COUNT(*) FROM audio_files')
        audio_count = self.cursor.fetchone()[0]
        print(f"Database now has {audio_count} audio file entries.")

    def file_already_registered(self, file_path):
        """Checks if the audio file path already exists in the audio_files table."""
        self.cursor.execute('SELECT 1 FROM audio_files WHERE file_path = ? LIMIT 1', (file_path,))
        return self.cursor.fetchone() is not None

    def register_audio(self, file_path, audio_info, fingerprints, algorithm_name):
        """
        Saves the audio file information and its fingerprints for a specific algorithm.
        Returns the audio_id if successful, None otherwise.
        """
        if not fingerprints:
            print(f"Warning: No fingerprints provided for {file_path} with algorithm {algorithm_name}. Skipping registration.")
            return None

        # Check if file exists, insert if not, get audio_id
        audio_id = None
        self.cursor.execute('SELECT audio_id FROM audio_files WHERE file_path = ?', (file_path,))
        result = self.cursor.fetchone()
        if result:
            audio_id = result[0]
            print(f"File {file_path} already exists with audio_id {audio_id}. Adding fingerprints for algorithm '{algorithm_name}'.")
        else:
            try:
                self.cursor.execute(
                    'INSERT INTO audio_files (file_path, filename) VALUES (?, ?)',
                    (file_path, audio_info.get('filename', os.path.basename(file_path)))
                )
                audio_id = self.cursor.lastrowid
                print(f"Registered new file: {file_path} with audio_id {audio_id}")
            except sqlite3.Error as e:
                print(f"Database error inserting audio file {file_path}: {e}")
                self.conn.rollback()
                return None

        # Currently, we support registration for MaximaPairingAlgorithm only.
        if algorithm_name == "MaximaPairingAlgorithm":
            try:
                fingerprint_data = [
                    (hash_hex, anchor_time, audio_id)
                    for hash_hex, anchor_time in fingerprints
                ]
                self.cursor.executemany(
                    'INSERT INTO maxima_pairing_fingerprints (hash_hex, anchor_time, audio_id) VALUES (?, ?, ?)',
                    fingerprint_data
                )
                self.conn.commit()
                print(f"Successfully registered {len(fingerprint_data)} fingerprints for audio_id {audio_id} in table 'maxima_pairing_fingerprints'.")
                return audio_id

            except sqlite3.Error as e:
                print(f"Database error during fingerprint registration for {file_path} (algorithm {algorithm_name}): {e}")
                self.conn.rollback()
                return None
        else:
            print(f"Algorithm '{algorithm_name}' is not supported for registration.")
            return None

    def get_audio_info(self, audio_id):
        """Retrieves the audio file path and filename for a given audio_id."""
        self.cursor.execute('SELECT file_path, filename FROM audio_files WHERE audio_id = ?', (audio_id,))
        result = self.cursor.fetchone()
        if result:
            return {'path': result[0], 'filename': result[1]}
        print(f"Warning: Could not find audio info for audio_id {audio_id}")
        return None

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.commit()  # Ensure all changes are saved
            self.conn.close()
            print("Database connection closed.")
        else:
            print("No database connection to close.") 