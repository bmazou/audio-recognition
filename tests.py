import hashlib
import os
import sys
import time

import librosa

from chroma_algorithm import ChromaAlgorithm
from maxima_pairing_algorithm import MaximaPairingAlgorithm
from spectral_patch_algorithm import SpectralPatchAlgorithm
from sqlite_db import SQLiteDB


def get_all_audio_files(data_dir, exts={'.wav', '.mp3', '.flac', '.ogg', '.m4a'}):
    """
    Walk through the given directory and return a list of file paths
    that match the provided audio extensions.
    """
    audio_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                audio_files.append(os.path.join(root, f))
    return audio_files


def test_algorithm(algorithm, db, file_path):
    """
    Generate fingerprints for given file using algorithm, find a match in the DB
    then verify if the retrieved match corresponds to the current file.
    """
    fingerprints = algorithm.generate_fingerprints(file_path)
    if not fingerprints:
        print(f"ERROR: No fingerprints produced for {file_path}.")
        return False

    best_match_audio_id, message = algorithm.find_match(fingerprints, db)
    if best_match_audio_id is None:
        print(f"ERROR: Matching failed for {file_path}: {message}")
        return False

    audio_info = db.get_audio_info(best_match_audio_id)
    if not audio_info:
        print(f"ERROR: Audio info for audio_id {best_match_audio_id} not found.")
        return False

    registered_path = os.path.abspath(audio_info['path'])
    current_path = os.path.abspath(file_path)
    if registered_path == current_path:
        return True
    else:
        return False


def register_audio_files(algo, db, audio_files):
    """
    Register audio files in the database using the provided algorithm.
    Returns total fingerprints and total duration.
    """
    total_fingerprints = 0
    total_duration = 0.0

    print(f"INFO: Registering {len(audio_files)} audio files")
    for i,file_path in enumerate(audio_files):
        if i % 500 == 0 and i != 0:
            print(f"INFO: Registered {i}/{len(audio_files)} files")

        try:
            fingerprints = algo.generate_fingerprints(file_path)
            if fingerprints:
                total_fingerprints += len(fingerprints)
                duration = librosa.get_duration(path=file_path)
                total_duration += duration
                
        except MemoryError:
            print(f"ERROR: MemoryError processing file: {file_path}. Skipping.")
            continue
        except Exception as e:
            print(f"ERROR: Error processing file {file_path}: {e}. Skipping.")
            continue

        db.register_audio(file_path, {"filename": os.path.basename(file_path)}, fingerprints, algo.name)
        
    return total_fingerprints, total_duration

def test_algo(algo, clear_db):
    """ Test the algorithm by registering audio files and then matching them."""
    start_time = time.time()

    print(f"INFO: Testing {algo.name} algorithm")

    data_dir = "data"
    audio_files = get_all_audio_files(data_dir)
    if not audio_files:
        print(f"ERROR: No audio files found in {data_dir}")
        sys.exit(1)

    db_path = "fingerprints.db"
    db = SQLiteDB(db_path=db_path, clear_db=clear_db)

    total_fingerprints, total_duration = register_audio_files(algo, db, audio_files)

    register_time = time.time() - start_time
    print(f"INFO: Registration completed in {register_time:.2f} seconds")
    if total_duration:
        print(f"INFO: Average fingerprints per second: {total_fingerprints/total_duration:.2f}")

    # Testing part:
    test_start = time.time()
    total_tests = len(audio_files)
    passed_tests = 0

    for i, file_path in enumerate(audio_files):
        if i % 500 == 0 and i != 0:
            print(f"INFO: Processing file {i}/{len(audio_files)}")
        
        if test_algorithm(algo, db, file_path):
            passed_tests += 1

    print("\n\n\n")
    print(f"INFO: Matching took: {time.time() - test_start:.2f} seconds")
    print(f"INFO: SUMMARY: {passed_tests}/{total_tests} ({passed_tests / total_tests:.2%}) tests passed")
    print("-"*50)
    db.close()


def test_maxima_pairing_algo(sr=22050, n_fft=1024, hop_length=512,
                               neighborhood_size=20, min_amplitude=-20,
                               target_t_min=5, target_t_max=40, target_f_max_delta=100,clear_db=True):
    """Test the MaximaPairingAlgorithm with given parameters."""
    algo = MaximaPairingAlgorithm(
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        neighborhood_size=neighborhood_size,
        min_amplitude=min_amplitude,
        target_t_min=target_t_min,
        target_t_max=target_t_max,
        target_f_max_delta=target_f_max_delta,
        hash_algorithm=hashlib.sha1
    )

    test_algo(algo, clear_db=clear_db)
    
    
def test_chroma_algo(threshold=0.5, sr=22050, n_fft=2048, hop_length=512, clear_db=True):
    """Test the ChromaAlgorithm with given parameters."""
    algo = ChromaAlgorithm(
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        threshold=threshold,
        hash_algorithm=hashlib.sha1
    )
    
    test_algo(algo, clear_db=clear_db)


def main():
    test_maxima_pairing_algo()
    # test_chroma_algo()

if __name__ == '__main__':
    main()