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

2. Fingerprint tables:
    a. maxima_pairing_fingerprints table:
        - hash_hex TEXT NOT NULL
        - anchor_time INTEGER NOT NULL
        - audio_id INTEGER NOT NULL
        - FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id)
    b. spectral_patch_fingerprints table:
        - hash_hex TEXT NOT NULL
        - patch_time INTEGER NOT NULL
        - audio_id INTEGER NOT NULL
        - FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id)

Indices:
- On maxima_pairing_fingerprints(hash_hex)
- On spectral_patch_fingerprints(hash_hex)
- On audio_files(file_path)
"""

class SQLiteDB:
    def __init__(self, db_path='fingerprints.db', clear_db=False):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        if clear_db:
            self._clear_db()
        self._create_tables()

    def _create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS audio_files (
                audio_id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL
            )
        ''')
        
        # Table for MaximaPairingAlgorithm fingerprints.
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

        # Table for SpectralPatchAlgorithm fingerprints.
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS spectral_patch_fingerprints (
                hash_hex TEXT NOT NULL,
                patch_time INTEGER NOT NULL,
                audio_id INTEGER NOT NULL,
                FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id) ON DELETE CASCADE
            )
        ''')
        
        # Table for ChromaAlgorithm fingerprints.
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS chroma_fingerprints (
                hash_hex TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                audio_id INTEGER NOT NULL,
                FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id)
            )
        ''')
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_chroma_hash
            ON chroma_fingerprints (hash_hex)
        ''')
        
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_spectral_patch_hash
            ON spectral_patch_fingerprints (hash_hex)
        ''')
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_audio_files_path
            ON audio_files (file_path)
        ''')
        self.conn.commit()

    def _clear_db(self):
        print(f"Clearing database '{self.db_path}'...")
        self.cursor.execute('DROP TABLE IF EXISTS maxima_pairing_fingerprints')
        self.cursor.execute('DROP TABLE IF EXISTS spectral_patch_fingerprints')
        self.cursor.execute('DROP TABLE IF EXISTS audio_files')
        self.conn.commit()
        self._create_tables()
        print("Database cleared and schema recreated.")
        self.cursor.execute('SELECT COUNT(*) FROM audio_files')
        audio_count = self.cursor.fetchone()[0]
        print(f"Database now has {audio_count} audio file entries.")
    
    def fingerprint_already_registered(self, file_path, algorithm_name):
        if algorithm_name == "MaximaPairingAlgorithm":
            self.cursor.execute('''
                SELECT 1 FROM maxima_pairing_fingerprints
                JOIN audio_files ON audio_files.audio_id = maxima_pairing_fingerprints.audio_id
                WHERE audio_files.file_path = ? LIMIT 1
            ''', (file_path,))
            
        elif algorithm_name == "SpectralPatchAlgorithm":
            self.cursor.execute('''
                SELECT 1 FROM spectral_patch_fingerprints
                JOIN audio_files ON audio_files.audio_id = spectral_patch_fingerprints.audio_id
                WHERE audio_files.file_path = ? LIMIT 1
            ''', (file_path,))
            
        elif algorithm_name == "ChromaAlgorithm":
            self.cursor.execute('''
                SELECT 1 FROM chroma_fingerprints
                JOIN audio_files ON audio_files.audio_id = chroma_fingerprints.audio_id
                WHERE audio_files.file_path = ? LIMIT 1
            ''', (file_path,))
        
        else:
            print(f"Unknown algorithm name: {algorithm_name}")
            return False

        return self.cursor.fetchone() is not None

    def register_audio(self, file_path, audio_info, fingerprints, algorithm_name):
        if not fingerprints:
            print(f"Warning: No fingerprints provided for {file_path} with algorithm {algorithm_name}. Skipping registration.")
            return None

        audio_id = None
        self.cursor.execute('SELECT audio_id FROM audio_files WHERE file_path = ?', (file_path,))
        result = self.cursor.fetchone()
        if result:
            audio_id = result[0]
            print(f"File {file_path} already exists with audio_id {audio_id}. Adding fingerprints for '{algorithm_name}'.")
        else:
            try:
                self.cursor.execute(
                    'INSERT INTO audio_files (file_path, filename) VALUES (?, ?)',
                    (file_path, audio_info.get('filename', os.path.basename(file_path)))
                )
                audio_id = self.cursor.lastrowid
                # print(f"Registered new file: {file_path} with audio_id {audio_id}")
            except sqlite3.Error as e:
                print(f"Database error inserting audio file {file_path}: {e}")
                self.conn.rollback()
                return None

        # Register fingerprints based on the selected algorithm.
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
                # print(f"Registered {len(fingerprint_data)} fingerprints in 'maxima_pairing_fingerprints' for audio_id {audio_id}.")
                
                return audio_id
            except sqlite3.Error as e:
                print(f"Database error during registration (MaximaPairingAlgorithm) for {file_path}: {e}")
                self.conn.rollback()
                return None
            
        elif algorithm_name == "SpectralPatchAlgorithm":
            try:
                fingerprint_data = [
                    (hash_hex, patch_time, audio_id)
                    for hash_hex, patch_time in fingerprints
                ]
                self.cursor.executemany(
                    'INSERT INTO spectral_patch_fingerprints (hash_hex, patch_time, audio_id) VALUES (?, ?, ?)',
                    fingerprint_data
                )
                self.conn.commit()
                print(f"Registered {len(fingerprint_data)} fingerprints in 'spectral_patch_fingerprints' for audio_id {audio_id}.")
                return audio_id
            except sqlite3.Error as e:
                print(f"Database error during registration (SpectralPatchAlgorithm) for {file_path}: {e}")
                self.conn.rollback()
                return None
            
        elif algorithm_name == "ChromaAlgorithm":
            try:
                fingerprint_data = [
                    (hash_hex, frame_index, audio_id)
                    for hash_hex, frame_index in fingerprints
                ]
                self.cursor.executemany(
                    'INSERT INTO chroma_fingerprints (hash_hex, frame_index, audio_id) VALUES (?, ?, ?)',
                    fingerprint_data
                )
                self.conn.commit()
                print(f"Registered {len(fingerprint_data)} fingerprints in 'chroma_fingerprints' for audio_id {audio_id}.")
                return audio_id
            except sqlite3.Error as e:
                print(f"Database error during registration (ChromaAlgorithm) for {file_path}: {e}")
                self.conn.rollback()
                return None
            
        else:
            print(f"Algorithm '{algorithm_name}' is not supported for registration.")
            return None


    def get_audio_info(self, audio_id):
        self.cursor.execute('SELECT file_path, filename FROM audio_files WHERE audio_id = ?', (audio_id,))
        result = self.cursor.fetchone()
        if result:
            return {'path': result[0], 'filename': result[1]}
        print(f"Warning: Could not find audio info for audio_id {audio_id}")
        return None

    def close(self):
        if self.conn:
            self.conn.commit()
            self.conn.close()
            print("Database connection closed.")
        else:
            print("No database connection to close.")