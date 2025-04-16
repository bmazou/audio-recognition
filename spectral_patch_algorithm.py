import numpy as np

from fingerprint_algorithm import FingerprintAlgorithm


class SpectralPatchAlgorithm(FingerprintAlgorithm):
    ALGORITHM_NAME = "SpectralPatchAlgorithm"

    def __init__(self, sr, n_fft, hop_length, patch_size, min_patch_energy, hash_algorithm):
        super().__init__(self.ALGORITHM_NAME)
        self.sr = sr                              # Target sample rate
        self.n_fft = n_fft                        # Window size for FFT
        self.hop_length = hop_length              # Number of samples between successive frames
        self.patch_size = patch_size              # Size of the patch (for both frequency and time)
        self.min_patch_energy = min_patch_energy  # Minimum average energy in a patch to be considered
        self.hash_algorithm = hash_algorithm      # Hash algorithm 

        # Note: for simplicity, we use a single `patch_size` for both dimension. 
        #   `patch_size_freq` and `patch_size_time` could also be used.

    def generate_fingerprints(self, file_path, start_time=None, end_time=None):
        audio = self._load_and_preprocess_audio(file_path, self.sr, start_time, end_time)
        if audio is None:
            print(f"Error loading audio from {file_path}")
            return None

        spectrogram = self._calculate_spectrogram(audio, self.n_fft, self.hop_length)
        num_freq_bins, num_frames = spectrogram.shape
        fingerprints = []

        # Slide over the spectrogram using "patch_size" steps
        for f in range(0, num_freq_bins-self.patch_size+1, self.patch_size):
            for t in range(0, num_frames-self.patch_size + 1, self.patch_size):
                patch = spectrogram[f:f+self.patch_size, t:t+self.patch_size]
                avg_energy = np.mean(patch)
                if avg_energy >= self.min_patch_energy:
                    
                    # Compute hash of the (flattened) patch
                    patch_flat = patch.flatten()
                    hash_input = patch_flat.tobytes()
                    hasher = self.hash_algorithm()
                    hasher.update(hash_input)
                    hash_hex = hasher.hexdigest()
                    
                    # `t` is the patch's time index (in spectrogram frames)
                    fingerprints.append((hash_hex, t))

        return fingerprints

    def find_match(self, query_fingerprints, db):
        """
        Finds the best match for the given query_fingerprints using data stored in 'spectral_patch_fingerprints' table.
        """
        if not query_fingerprints:
            return None, "No query fingerprints provided for SpectralPatchFingerprintAlgorithm."

        unique_query_hashes = {fp[0] for fp in query_fingerprints}
        print(f"Querying {len(unique_query_hashes)} unique hashes for algorithm '{self.ALGORITHM_NAME}'")

        placeholders = ','.join('?' for _ in unique_query_hashes)
        sql = f'''
            SELECT hash_hex, audio_id, patch_time
            FROM spectral_patch_fingerprints
            WHERE hash_hex IN ({placeholders})
        '''
        cursor = db.conn.cursor()
        try:
            cursor.execute(sql, list(unique_query_hashes))
            results = cursor.fetchall()
        except Exception as e:
            print(f"Database error during lookup for algorithm {self.ALGORITHM_NAME}: {e}")
            return None, "Database error during search."

        print(f"Retrieved {len(results)} hash matches from DB for algorithm '{self.ALGORITHM_NAME}'.")

        db_matches_by_hash = {}
        for db_hash, audio_id, db_patch_time in results:
            db_matches_by_hash.setdefault(db_hash, []).append((audio_id, db_patch_time))

        potential_matches = {}
        for query_hash, query_time in query_fingerprints:
            if query_hash not in db_matches_by_hash: continue
            
            for audio_id, db_patch_time in db_matches_by_hash[query_hash]:
                potential_matches.setdefault(audio_id, []).append((db_patch_time, query_time))

        if not potential_matches:
            return None, f"No matching hashes found in the database for algorithm '{self.ALGORITHM_NAME}'."

        # Score each audio based on consistent time offset between matching patches.
        match_scores = {}
        for audio_id, time_pairs in potential_matches.items():
            delta_counts = {}
            for db_time, query_time in time_pairs:
                delta = db_time - query_time
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