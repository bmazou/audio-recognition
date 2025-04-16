import hashlib
from abc import ABC, abstractmethod

import librosa
import numpy as np


class FingerprintAlgorithm(ABC):
    """Abstract base class for audio fingerprinting algorithms."""

    def __init__(self, algorithm_name):
        """
        Basic initialization to store algorithm name
        Specific parameters are handled by subclass __init__.
        """
        
        self._algorithm_name = algorithm_name

    @property
    def name(self):
        """Returns the name of the algorithm."""
        return self._algorithm_name

    @abstractmethod
    def generate_fingerprints(self, file_path, start_time=None, end_time = None):
        """Generates fingerprints for the given audio file (or its segment)."""
        pass
    
    @abstractmethod
    def find_match(self, query_fingerprints, db):
        """Finds matches for the given fingerprints in the database."""
        pass 
    
    def _calculate_spectrogram(self, y, n_fft, hop_length):
        """Calculates the magnitude spectrogram using short-time Fourier transform (STFT)."""
        stft_result = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        spectrogram = np.abs(stft_result)
        return spectrogram

    def _load_and_preprocess_audio(self, file_path, target_sr):
        """Loads an audio file, ensures it is mono, and resamples to the target sample rate"""
        try:
            audio, original_sr = librosa.load(file_path, sr=None, mono=False)
            # Convert to mono if necessary.
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)
            
            # Resample if the original sample rate is different.
            if original_sr != self.sr:
                audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)

            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def _cut_audio(self, audio, start_time, end_time, sr):
        """Cuts the audio to the specified time range. Returns original audio if no valid range is provided."""
        if start_time is None or end_time is None:
            return audio
        
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

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