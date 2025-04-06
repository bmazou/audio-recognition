import librosa
import numpy as np
from scipy.ndimage import maximum_filter


class FingerprintGenerator:
    def __init__(self, sr, n_fft, hop_length, neighborhood_size, min_amplitude, 
                 target_t_min, target_t_max, target_f_max_delta, hash_algorithm):
        self.sr = sr                                    # Target sample rate (Hz)
        self.n_fft = n_fft                              # Window size for FTT (number of samples)
        self.hop_length = hop_length                    # Number of samples between successive frames
        self.neighborhood_size = neighborhood_size      # Size of the neighborhood for peak detection
        self.min_amplitude = min_amplitude              # Minimum amplitude for peak detection
        self.target_t_min = target_t_min                # Minimum time difference for fingerprint pairs (in STFT frames)
        self.target_t_max = target_t_max                # Maximum time difference for fingerprint pairs (in STFT frames)
        self.target_f_max_delta = target_f_max_delta    # Maximum frequency difference for fingerprint pairs (in frequency bins)
        self.hash_algorithm = hash_algorithm            # Hash algorithm for fingerprint generation 
        self.audio_id = -1                              # Autoincrementing ID for each audio file

    def load_and_preprocess_audio(self, file_path):
        """Loads an audio file, ensures it is mono, and resamples to the target sample rate"""
        try:
            y, original_sr = librosa.load(file_path, sr=None, mono=False)
            # Convert to mono if necessary.
            if y.ndim > 1:
                y = np.mean(y, axis=0)
            # Resample if the original sample rate is different.
            if original_sr != self.sr:
                y = librosa.resample(y, orig_sr=original_sr, target_sr=self.sr)
            return y
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def calculate_spectrogram(self, y):
        """Calculates the magnitude spectrogram using short-time Fourier transform (STFT)."""
        stft_result = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        spectrogram = np.abs(stft_result)
        return spectrogram

    def find_spectrogram_peaks(self, spectrogram):
        """Uses a maximum filter to detect local peaks in the spectrogram."""
        data_max = maximum_filter(spectrogram, size=self.neighborhood_size, mode='constant', cval=0.0)
        peaks_mask = (spectrogram == data_max)
        peaks_mask &= (spectrogram >= self.min_amplitude)
        peak_coords = np.argwhere(peaks_mask)
        return peak_coords  # Each peak is [frequency_bin, time_frame]

    def generate_fingerprints(self, file_path, is_query=False):
        if not is_query:
            self.audio_id += 1            

        y = self.load_and_preprocess_audio(file_path)

        error_loading_file = y is None
        if error_loading_file:
            return None, None

        spectrogram = self.calculate_spectrogram(y)
        peaks = self.find_spectrogram_peaks(spectrogram)
        if peaks.size == 0:
            return None, None
        
        
        fingerprints = []
        peaks = sorted(peaks, key=lambda p: (p[1], p[0]))
        num_peaks = len(peaks)

        for i in range(num_peaks):
            anchor_freq, anchor_time = peaks[i]
            target_min_time = anchor_time + self.target_t_min
            target_max_time = anchor_time + self.target_t_max

            for j in range(i + 1, num_peaks):
                target_freq, target_time = peaks[j]

                if target_time < target_min_time:
                    continue
                if target_time > target_max_time:
                    break

                if abs(target_freq - anchor_freq) <= self.target_f_max_delta:
                    time_delta = target_time - anchor_time
                    hash_input = f"{anchor_freq}:{target_freq}:{time_delta}".encode('utf-8')
                    hasher = self.hash_algorithm()
                    hasher.update(hash_input)
                    hash_hex = hasher.hexdigest()
                    fingerprints.append((hash_hex, anchor_time, self.audio_id))
                    
        
                    
        return fingerprints, self.audio_id