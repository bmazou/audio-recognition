import librosa
import numpy as np

from fingerprint_algorithm import FingerprintAlgorithm


class ChromaAlgorithm(FingerprintAlgorithm):
    ALGORITHM_NAME = "ChromaAlgorithm"

    def __init__(self, sr, n_fft, hop_length, threshold, hash_algorithm):
        """Initializes the ChromaAlgorithm with specific parameters."""
        super().__init__(self.ALGORITHM_NAME)
        self.sr = sr                              # Target sample rate
        self.n_fft = n_fft                        # Window size for FFT
        self.hop_length = hop_length              # Number of samples between frames
        self.threshold = threshold                # Minimum chroma intensity for fingerprinting
        self.hash_algorithm = hash_algorithm      # Hash algorithm

    def generate_fingerprints(self, file_path, start_time=None, end_time=None):
        """Generates fingerprints based on dominant chroma features in audio frames."""
        audio = self._load_and_preprocess_audio(file_path, self.sr, start_time, end_time)
        if audio is None:
            print(f"Error loading audio from {file_path}")
            return None

        # Compute the chroma feature (chromagram)
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        fingerprints = []
        num_frames = chroma.shape[1]
        
        # For each time frame, if the max chroma intensity exceeds the threshold,
        # generate a fingerprint based on the dominant chroma bin and the frame index
        for t in range(num_frames):
            frame = chroma[:, t]
            max_val = np.max(frame)
            if max_val >= self.threshold:
                dominant_bin = int(np.argmax(frame))
                hash_input = f"{dominant_bin}:{t}".encode('utf-8')
                hasher = self.hash_algorithm()
                hasher.update(hash_input)
                hash_hex = hasher.hexdigest()
                fingerprints.append((hash_hex, t))
        
        if not fingerprints:
            print(f"No fingerprints generated for {file_path} with ChromaAlgorithm.")
        return fingerprints

    def find_match(self, query_fingerprints, db):
        """Finds the best match for chroma-based query fingerprints in the database."""
        if not query_fingerprints:
            return None, "No query fingerprints provided for ChromaAlgorithm."

        unique_query_hashes = {fp[0] for fp in query_fingerprints}

        placeholders = ','.join('?' for _ in unique_query_hashes)
        sql = f'''
            SELECT hash_hex, audio_id, frame_index
            FROM chroma_fingerprints
            WHERE hash_hex IN ({placeholders})
        '''
        cursor = db.conn.cursor()
        try:
            cursor.execute(sql, list(unique_query_hashes))
            results = cursor.fetchall()
        except Exception as e:
            print(f"Database error during lookup for algorithm {self.ALGORITHM_NAME}: {e}")
            return None, "Database error during search."


        db_matches_by_hash = {}
        for db_hash, audio_id, db_frame_index in results:
            db_matches_by_hash.setdefault(db_hash, []).append((audio_id, db_frame_index))

        potential_matches = {}
        for query_hash, query_frame in query_fingerprints:
            if query_hash in db_matches_by_hash:
                for audio_id, db_frame_index in db_matches_by_hash[query_hash]:
                    potential_matches.setdefault(audio_id, []).append((db_frame_index, query_frame))

        if not potential_matches:
            return None, f"No matching hashes found in the database for algorithm '{self.ALGORITHM_NAME}'."

        match_scores = {}
        for audio_id, time_pairs in potential_matches.items():
            delta_counts = {}
            for db_frame, query_frame in time_pairs:
                delta = db_frame - query_frame
                delta_counts[delta] = delta_counts.get(delta, 0) + 1
            if delta_counts:
                best_score = max(delta_counts.values())
                match_scores[audio_id] = best_score

        if not match_scores:
            return None, "Matching hashes found, but could not determine a consistent time alignment."

        best_match_audio_id = max(match_scores, key=match_scores.get)
        best_score = match_scores[best_match_audio_id]
        audio_info = db.get_audio_info(best_match_audio_id)
        if not audio_info:
            return None, f"Match found (Audio ID: {best_match_audio_id}, Score: {best_score}), but could not retrieve audio details."
        audio_name = audio_info['filename']
        return best_match_audio_id, f"Best match: '{audio_name}' (ID: {best_match_audio_id}). Score: {best_score}." 