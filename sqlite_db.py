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

2. algorithms table:
    - algorithm_id INTEGER PRIMARY KEY AUTOINCREMENT
    - name TEXT UNIQUE NOT NULL
    
3. fingerprints table:
   - hash_hex TEXT NOT NULL
   - anchor_time INTEGER NOT NULL
   - audio_id INTEGER NOT NULL
   - algorithm_id INTEGER NOT NULL
   - FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id)
   - FOREIGN KEY (algorithm_id) REFERENCES algorithms(algorithm_id)

Indices:
- On fingerprints(hash_hex) for faster lookup during matching
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
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS algorithms (
                algorithm_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS fingerprints (
                hash_hex TEXT NOT NULL,
                anchor_time INTEGER NOT NULL,
                audio_id INTEGER NOT NULL,
                algorithm_id INTEGER NOT NULL,
                FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id) ON DELETE CASCADE,
                FOREIGN KEY (algorithm_id) REFERENCES algorithms(algorithm_id) ON DELETE CASCADE
            )
        ''')
        
        # Index for faster hash lookups by algorithm during matching
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_fingerprints_hash_algo
            ON fingerprints (hash_hex, algorithm_id)
        ''')
        
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_audio_files_path
            ON audio_files (file_path)
        ''')  # SQLite should automatically create index due to UNIQUE constraint, but just to be sure (and to be explicit)
        self.conn.commit()

    def _clear_db(self):
        """Removes all data from the database by dropping and recreating tables."""
        print(f"Clearing database '{self.db_path}'...")
        self.cursor.execute('DROP TABLE IF EXISTS fingerprints')
        self.cursor.execute('DROP TABLE IF EXISTS algorithms')
        self.cursor.execute('DROP TABLE IF EXISTS audio_files')
        self.conn.commit()
        self._create_tables() 
        print("Database cleared and schema recreated.")
        
        self.cursor.execute('SELECT COUNT(*) FROM audio_files')
        audio_count = self.cursor.fetchone()[0]
        self.cursor.execute('SELECT COUNT(*) FROM algorithms')
        algo_count = self.cursor.fetchone()[0]
        print(f"Database now has {audio_count} audio file entries and {algo_count} algorithm entries.")

    def _get_or_create_algorithm_id(self, algorithm_name):
        """Gets the ID for an algorithm name, creating it if it doesn't exist yet."""
        self.cursor.execute('SELECT algorithm_id FROM algorithms WHERE name = ?', (algorithm_name,))
        result = self.cursor.fetchone()
        if result:
            return result[0]
        else:
            self.cursor.execute('INSERT INTO algorithms (name) VALUES (?)', (algorithm_name,))
            algo_id = self.cursor.lastrowid
            self.conn.commit()      
            print(f"Registered new algorithm: '{algorithm_name}' with ID: {algo_id}")
            return algo_id


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

        algorithm_id = self._get_or_create_algorithm_id(algorithm_name)
        if algorithm_id is None:
            print(f"Error: Could not get or create algorithm ID for '{algorithm_name}'. Can't register fingerprints.")
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

        # Insert fingerprints
        try:
            fingerprint_data = [
                (hash_hex, anchor_time, audio_id, algorithm_id)
                for hash_hex, anchor_time in fingerprints
            ]

            self.cursor.executemany(
                'INSERT INTO fingerprints (hash_hex, anchor_time, audio_id, algorithm_id) VALUES (?, ?, ?, ?)',
                fingerprint_data
            )
            self.conn.commit()
            print(f"Successfully registered {len(fingerprint_data)} fingerprints for audio_id {audio_id}, algorithm_id {algorithm_id}.")
            return audio_id

        except sqlite3.Error as e:
            print(f"Database error during fingerprint registration for {file_path} (algorithm {algorithm_name}): {e}")
            self.conn.rollback()
            return None


    def _score_potential_matches(self, potential_matches):
        """Scores potential matches based on time-difference alignment."""
        match_scores = defaultdict(lambda: defaultdict(int)) # {audio_id: {delta: count}}
        final_scores = {} # {audio_id: best_score}

        for audio_id, time_pairs in potential_matches.items():
            for db_time, query_time in time_pairs:
                db_time_int = int.from_bytes(db_time, byteorder=sys.byteorder)
                delta = db_time_int - int(query_time) 
                match_scores[audio_id][delta] += 1

            if match_scores[audio_id]:  # Check if audio_id has any deltas
                # Find the delta with the highest count for this audio_id
                best_delta = max(match_scores[audio_id], key=match_scores[audio_id].get)
                final_scores[audio_id] = match_scores[audio_id][best_delta]

        return final_scores

    def find_match(self, query_fingerprints, algorithm_name):
        """
        Finds the best match for a list of query fingerprints generated by a specific algorithm.
        Returns (best_match_audio_id, message) or (None, message).
        """
        if not query_fingerprints:
             return None, "No query fingerprints provided."

        # Look up the algorithm ID - do not create it here
        self.cursor.execute('SELECT algorithm_id FROM algorithms WHERE name = ?', (algorithm_name,))
        result = self.cursor.fetchone()
        if not result:
            return None, f"Algorithm '{algorithm_name}' not found in the database. Cannot perform match."
        algorithm_id = result[0]

        # `potential_matches` stores {db_audio_id: [(db_anchor_time, query_anchor_time), ...]}
        potential_matches = defaultdict(list)

        unique_query_hashes = {fp[0] for fp in query_fingerprints}
        print(f"Querying {len(unique_query_hashes)} unique hashes for algorithm '{algorithm_name}'")

        placeholders = ','.join('?' for _ in unique_query_hashes)
        # Query fingerprints matching the hash AND the specific algorithm ID
        sql = f'''
            SELECT hash_hex, audio_id, anchor_time
            FROM fingerprints
            WHERE hash_hex IN ({placeholders}) AND algorithm_id = ?
        '''
        query_params = list(unique_query_hashes) + [algorithm_id]

        # Store results indexed by hash for efficient lookup
        db_matches_by_hash = defaultdict(list) # {hash: [(audio_id, anchor_time), ...]}
        try:
            self.cursor.execute(sql, query_params)
            results = self.cursor.fetchall()
            for db_hash, db_audio_id, db_anchor_time in results:
                db_matches_by_hash[db_hash].append((db_audio_id, db_anchor_time))
        except sqlite3.Error as e:
            print(f"Database error during hash lookup for algorithm {algorithm_name}: {e}")
            return None, "Database error during search."

        print(f"Retrieved {len(results)} total hash matches from DB for algorithm '{algorithm_name}'.")

        # Correlate query fingerprints with database matches
        for query_hash, query_anchor_time in query_fingerprints:
            if query_hash in db_matches_by_hash:
                for db_audio_id, db_anchor_time in db_matches_by_hash[query_hash]:
                    potential_matches[db_audio_id].append((db_anchor_time, query_anchor_time))

        if not potential_matches:
            return None, f"No matching hashes found in the database for algorithm '{algorithm_name}'."

        print(f"Found potential matches in {len(potential_matches)} audio files. Scoring...")

        final_scores = self._score_potential_matches(potential_matches)

        if not final_scores:
            return None, "Matching hashes found, but could not determine a consistent time alignment."

        # Find the audio_id with the highest score
        best_match_audio_id = max(final_scores, key=final_scores.get)
        best_score = final_scores[best_match_audio_id]
        audio_info = self._get_audio_info(best_match_audio_id)

        if not audio_info:
             return None, f"Match found (Audio ID: {best_match_audio_id}, Score: {best_score}), but failed to retrieve audio file details."

        audio_name = audio_info['filename']
        return best_match_audio_id, f"Best match: '{audio_name}' (ID: {best_match_audio_id}). Score: {best_score}."

    def _get_audio_info(self, audio_id):
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