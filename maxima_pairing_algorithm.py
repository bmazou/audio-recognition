import hashlib
import sys

import librosa
import numpy as np
from scipy.ndimage import maximum_filter

from fingerprint_algorithm import FingerprintAlgorithm


class MaximaPairingAlgorithm(FingerprintAlgorithm):
    ALGORITHM_NAME = "MaximaPairingAlgorithm"
    
    def __init__(self, sr, n_fft, hop_length, neighborhood_size, min_amplitude,
                 target_t_min, target_t_max, target_f_max_delta, hash_algorithm):
        super().__init__(self.ALGORITHM_NAME)
        self.sr = sr                                    # Target sample rate (Hz)
        self.n_fft = n_fft                              # Window size for FTT (number of samples)
        self.hop_length = hop_length                    # Number of samples between successive frames
        self.neighborhood_size = neighborhood_size      # Size of the neighborhood for peak detection
        self.min_amplitude = min_amplitude              # Minimum amplitude for peak detection
        self.target_t_min = target_t_min                # Minimum time difference for fingerprint pairs (in STFT frames)
        self.target_t_max = target_t_max                # Maximum time difference for fingerprint pairs (in STFT frames)
        self.target_f_max_delta = target_f_max_delta    # Maximum frequency difference for fingerprint pairs (in frequency bins)
        self.hash_algorithm = hash_algorithm            # Hash algorithm for fingerprint generation 


    def _find_spectrogram_peaks(self, spectrogram):
        """Uses a maximum filter to detect local peaks in the spectrogram."""
        data_max = maximum_filter(spectrogram, size=self.neighborhood_size, mode='constant', cval=0.0)
        peaks_mask = (spectrogram == data_max)
        peaks_mask &= (spectrogram >= self.min_amplitude)
        peak_coords = np.argwhere(peaks_mask)
        return peak_coords  

    def generate_fingerprints(self, file_path, start_time=None, end_time=None):
        audio = self._load_and_preprocess_audio(file_path, self.sr)
        audio = self._cut_audio(audio, start_time, end_time, self.sr)
        
        if audio is None:
            print(f"Error loading audio from {file_path}")
            return None

        spectrogram = self._calculate_spectrogram(audio, self.n_fft, self.hop_length)
        peaks = self._find_spectrogram_peaks(spectrogram)
        if peaks.size == 0:
            print(f"No peaks found for {file_path}")
            return None 

        fingerprints = []
        peaks = sorted(peaks, key=lambda p: (p[1], p[0]))   # Sort by time, then frequency
        num_peaks = len(peaks)

        for i in range(num_peaks):
            anchor_freq, anchor_time = peaks[i]
            target_min_time = anchor_time + self.target_t_min
            target_max_time = anchor_time + self.target_t_max

            for j in range(i + 1, num_peaks):
                target_freq, target_time = peaks[j]

                if target_time > target_max_time:
                    break       # Moves to the next anchor

                time_delta = target_time - anchor_time
                if self.target_t_min <= time_delta <= self.target_t_max:
                    if abs(target_freq - anchor_freq) <= self.target_f_max_delta:
                        hash_input = f"{anchor_freq}:{target_freq}:{time_delta}".encode('utf-8')
                        hasher = self.hash_algorithm()
                        hasher.update(hash_input)
                        hash_hex = hasher.hexdigest()
                        fingerprints.append((hash_hex, anchor_time))

        return fingerprints  

    def _score_potential_matches(self, potential_matches):
        """
        Scores potential matches based on time-difference alignment.
        potential_matches is a dict {audio_id: [(db_anchor_time, query_anchor_time), ...]}
        Returns a dict {audio_id: best_score}
        """
        match_scores = {}  # {audio_id: score}
        for audio_id, time_pairs in potential_matches.items():
            delta_counts = {}
            for db_time, query_time in time_pairs:
                db_time_int = int.from_bytes(db_time, byteorder=sys.byteorder)
                delta = db_time_int - int(query_time) 
                delta_counts[delta] = delta_counts.get(delta, 0) + 1
            if delta_counts:
                best_score = max(delta_counts.values())
                match_scores[audio_id] = best_score
        return match_scores

    def find_match(self, query_fingerprints, db):
        """
        Finds the best match from the database for the given query_fingerprints.
        Returns (best_match_audio_id, message) or (None, message).
        """
        if not query_fingerprints:
            return None, "No query fingerprints provided."

        unique_query_hashes = {fp[0] for fp in query_fingerprints}
        print(f"Querying {len(unique_query_hashes)} unique hashes for algorithm '{self.ALGORITHM_NAME}'")

        placeholders = ','.join('?' for _ in unique_query_hashes)
        sql = f'''
            SELECT hash_hex, audio_id, anchor_time
            FROM maxima_pairing_fingerprints
            WHERE hash_hex IN ({placeholders})
        '''
        cursor = db.conn.cursor()
        try:
            cursor.execute(sql, list(unique_query_hashes))
            results = cursor.fetchall()
        except Exception as e:
            print(f"Database error during hash lookup for algorithm {self.ALGORITHM_NAME}: {e}")
            return None, "Database error during search."

        print(f"Retrieved {len(results)} total hash matches from DB for algorithm '{self.ALGORITHM_NAME}'.")

        # Group DB matches by hash for efficient correlating.
        db_matches_by_hash = {}
        for db_hash, audio_id, db_anchor_time in results:
            if db_hash not in db_matches_by_hash:
                db_matches_by_hash[db_hash] = []
            db_matches_by_hash[db_hash].append((audio_id, db_anchor_time))

        potential_matches = {}
        for query_hash, query_anchor_time in query_fingerprints:
            if query_hash in db_matches_by_hash:
                for audio_id, db_anchor_time in db_matches_by_hash[query_hash]:
                    potential_matches.setdefault(audio_id, []).append((db_anchor_time, query_anchor_time))

        if not potential_matches:
            return None, f"No matching hashes found in the database for algorithm '{self.ALGORITHM_NAME}'."

        print(f"Found potential matches in {len(potential_matches)} audio files. Scoring...")
        final_scores = self._score_potential_matches(potential_matches)

        if not final_scores:
            return None, "Matching hashes found, but could not determine a consistent time alignment."

        best_match_audio_id = max(final_scores, key=final_scores.get)
        best_score = final_scores[best_match_audio_id]
        audio_info = db.get_audio_info(best_match_audio_id)
        if not audio_info:
            return None, f"Match found (Audio ID: {best_match_audio_id}, Score: {best_score}), but failed to retrieve audio file details."

        audio_name = audio_info['filename']
        return best_match_audio_id, f"Best match: '{audio_name}' (ID: {best_match_audio_id}). Score: {best_score}."