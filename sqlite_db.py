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

2. fingerprints table:
   - hash_hex TEXT NOT NULL
   - anchor_time INTEGER NOT NULL
   - audio_id INTEGER NOT NULL
   - FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id)

Indices:
- On fingerprints(hash_hex) for faster lookup during matching
- On audio_files(file_path) for checking duplicates
"""

class SQLiteDB:
    def __init__(self, db_path='fingerprints.db'):
        """Initializes the connection and creates tables if they don't exist."""
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
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
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS fingerprints (
                hash_hex TEXT NOT NULL,
                anchor_time INTEGER NOT NULL,
                audio_id INTEGER NOT NULL,
                FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id) ON DELETE CASCADE
            )
        ''')
        
        # Index for faster hash lookups during matching
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_fingerprints_hash
            ON fingerprints (hash_hex)
        ''')
        # Index for faster file path lookups (though UNIQUE constraint helps)
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_audio_files_path
            ON audio_files (file_path)
        ''')  # SQLite should automatically create index due to UNIQUE constraint, but just to be sure (and to be explicit)
        self.conn.commit()

    def clear_db(self):
        """Removes all data from the database by dropping and recreating tables."""
        print(f"Clearing database '{self.db_path}'...")
        self.cursor.execute('DROP TABLE IF EXISTS fingerprints')
        self.cursor.execute('DROP TABLE IF EXISTS audio_files')
        self.conn.commit()
        self._create_tables() 
        print("Database cleared and schema recreated.")
        
        self.cursor.execute('SELECT COUNT(*) FROM audio_files')
        count = self.cursor.fetchone()[0]
        print(f"Database now has {count} audio file entries.")


    def file_already_registered(self, file_path):
        """Checks if the audio file path already exists in the database."""
        self.cursor.execute('SELECT 1 FROM audio_files WHERE file_path = ? LIMIT 1', (file_path,))
        return self.cursor.fetchone() is not None

    def register_audio(self, file_path, audio_info, fingerprints):
        """Saves the audio file information and its fingerprints in the database. Returns the generated audio_id."""
        
        if self.file_already_registered(file_path):
             print(f"Warning: Attempted to re-register already existing file: {file_path}")
             self.cursor.execute('SELECT audio_id FROM audio_files WHERE file_path = ?', (file_path,))
             result = self.cursor.fetchone()
             return result[0] if result else None

        try:
            # Insert audio file info and get the new audio_id
            self.cursor.execute(
                'INSERT INTO audio_files (file_path, filename) VALUES (?, ?)',
                (file_path, audio_info.get('filename', os.path.basename(file_path)))
            )
            audio_id = self.cursor.lastrowid    # id of the inserted row

            # Prepares fingerprint data for bulk insertion
            fingerprint_data = [
                (hash_hex, anchor_time, audio_id)
                for hash_hex, anchor_time in fingerprints
            ]

            self.cursor.executemany(
                'INSERT INTO fingerprints (hash_hex, anchor_time, audio_id) VALUES (?, ?, ?)',
                fingerprint_data
            )

            self.conn.commit()
            return audio_id

        except sqlite3.Error as e:
            print(f"Database error during registration of {file_path}: {e}")
            self.conn.rollback() 
            return None


    def _score_potential_matches(self, potential_matches):
        """Scores potential matches based on time-difference alignment."""

        match_scores = defaultdict(lambda: defaultdict(int))
        final_scores = {}

        for audio_id, time_pairs in potential_matches.items():
            for db_time, query_time in time_pairs:
                db_time_int = int.from_bytes(db_time, byteorder=sys.byteorder)
                delta = db_time_int - int(query_time)
                match_scores[audio_id][delta] += 1

            if match_scores[audio_id]:      # audio_id has some matches
                # Find the delta with the highest count for this audio_id
                best_delta = max(match_scores[audio_id], key=match_scores[audio_id].get)
                final_scores[audio_id] = match_scores[audio_id][best_delta]


        return final_scores


    def find_match(self, query_fingerprints):
        """Processes a list of query fingerprints and returns the best matching audio file id"""

        # `potential_matches` stores {db_audio_id: [(db_anchor_time, query_anchor_time), ...]}
        potential_matches = defaultdict(list)

        unique_query_hashes = {fp[0] for fp in query_fingerprints}
        print(f"Querying {len(unique_query_hashes)} unique hashes against the database...") # Debug


        placeholders = ','.join('?' for _ in unique_query_hashes)
        sql = f'SELECT hash_hex, audio_id, anchor_time FROM fingerprints WHERE hash_hex IN ({placeholders})'


        # Store results indexed by hash
        db_matches_by_hash = defaultdict(list)
        try:
            self.cursor.execute(sql, list(unique_query_hashes))
            results = self.cursor.fetchall()
            for db_hash, db_audio_id, db_anchor_time in results:
                db_matches_by_hash[db_hash].append((db_audio_id, db_anchor_time))
        except sqlite3.Error as e:
            print(f"Database error during hash lookup: {e}")
            return None, "Database error during search."

        print(f"Retrieved {len(results)} total hash matches from DB.")  


        for query_hash, query_anchor_time in query_fingerprints:
            if query_hash in db_matches_by_hash:
                
                # Adds all database matches for this hash to potential_matches
                for db_audio_id, db_anchor_time in db_matches_by_hash[query_hash]:
                    potential_matches[db_audio_id].append((db_anchor_time, query_anchor_time))


        if not potential_matches:
            return None, "No matching hashes found in the database."

        print(f"Found potential matches in {len(potential_matches)} audio files. Scoring...") # Debug


        final_scores = self._score_potential_matches(potential_matches)

        if not final_scores:
            return None, "Matching hashes found, but could not determine a consistent time alignment."

        # Find the audio_id with the highest score
        best_match_audio_id = max(final_scores, key=final_scores.get)
        best_score = final_scores[best_match_audio_id]
        audio_name = self._get_audio_info(best_match_audio_id)['filename']
        return best_match_audio_id, f"Best match: {audio_name}. Score: {best_score}."

    def _get_audio_info(self, audio_id):
        """Retrieves the audio file path and filename for a given audio_id."""
        self.cursor.execute('SELECT file_path, filename FROM audio_files WHERE audio_id = ?', (audio_id,))
        result = self.cursor.fetchone()
        if result:
            return {'path': result[0], 'filename': result[1]}
        return None

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.commit()  # Ensures all changes are saved
            self.conn.close()
            print("Database connection closed.")
        else:
            print("No database connection to close.")