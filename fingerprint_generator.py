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

    def _load_and_preprocess_audio(self, file_path):
        """Loads an audio file, ensures it is mono, and resamples to the target sample rate"""
        try:
            audio, original_sr = librosa.load(file_path, sr=None, mono=False)
            # Convert to mono if necessary.
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)
            
            # Resample if the original sample rate is different.
            if original_sr != self.sr:
                audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.sr)

            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def _calculate_spectrogram(self, y):
        """Calculates the magnitude spectrogram using short-time Fourier transform (STFT)."""
        stft_result = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        spectrogram = np.abs(stft_result)
        return spectrogram

    def _find_spectrogram_peaks(self, spectrogram):
        """Uses a maximum filter to detect local peaks in the spectrogram."""
        data_max = maximum_filter(spectrogram, size=self.neighborhood_size, mode='constant', cval=0.0)
        peaks_mask = (spectrogram == data_max)
        peaks_mask &= (spectrogram >= self.min_amplitude)
        peak_coords = np.argwhere(peaks_mask)
        return peak_coords  

    def _cut_audio(self, audio, start_time, end_time):
        """Cuts the audio to the specified time range. Returns original audio if no valid range is provided."""
        if start_time is None or end_time is None:
            return audio
        
        start_sample = int(start_time * self.sr)
        end_sample = int(end_time * self.sr)

        # Ensures the start and end times are valid
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        if start_sample >= end_sample:
            print(f"Invalid time range: start {start_time}s, end {end_time}s")
            return audio
        
        
        original_duration = len(audio) / self.sr
        cur_duration = len(audio[start_sample:end_sample]) / self.sr
        print(f"Cutting audio. Original duration: {original_duration:.2f}s, New duration: {cur_duration:.2f}s")
    
        return audio[start_sample:end_sample]
        


    def generate_fingerprints(self, file_path, start_time=None, end_time=None):
        audio = self._load_and_preprocess_audio(file_path)
        audio = self._cut_audio(audio, start_time, end_time)
        
        if audio is None:
            print(f"Error loading audio from {file_path}")
            return None

        spectrogram = self._calculate_spectrogram(audio)
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