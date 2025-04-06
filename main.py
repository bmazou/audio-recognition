import argparse
import hashlib
import os
import time
from collections import defaultdict

from fingerprint_generator import FingerprintGenerator
from redis_db import RedisDB

fingerprint_database = defaultdict(list)
audio_id_to_info = {}


def register_audio(file_path, audio_info, fingerprint_generator, redis_db):
    if redis_db.file_already_registered(file_path):
        print(f"Audio file '{file_path}' is already registered.")
        return
    
    
    start_time = time.time()
    fingerprints, audio_id = fingerprint_generator.generate_fingerprints(file_path)
    if not fingerprints:
        return

    redis_db.register_audio(file_path, audio_info, fingerprints, audio_id)
    # Register in Redis; if the file is already registered, the method returns the existing audio_id

    end_time = time.time()
    print(f"Registered {file_path}. Took {end_time - start_time:.2f}s.")

def find_match(query_file_path, fingerprint_generator, redis_db):
    """Processes a query audio file and attempts to find the best match using Redis."""
    print(f"\nQuerying file: {query_file_path}")
    start_time = time.time()

    query_fingerprints, _ = fingerprint_generator.generate_fingerprints(query_file_path, is_query=True)
    if not query_fingerprints:
        return None, "No fingerprints generated for query"

    print(f"Generated {len(query_fingerprints)} fingerprints.")

    best_match_audio_id, result_message = redis_db.find_match(query_fingerprints)
    if best_match_audio_id is not None:
        audio_info = redis_db.get_audio_info(best_match_audio_id)
        if audio_info:
            result_message += f"\n-- Match Info: {audio_info.get('filename', 'Unknown')}"
            
    end_time = time.time()
    result_message += f" -- Search took {end_time - start_time:.2f}s"
    return best_match_audio_id, result_message


def get_files(directory, extensions):
    """Recursively retrieves files in a directory matching the supported audio extensions."""
    print(f"Scanning directory '{directory}' for audio files...")
    files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            try:
                file_extension = os.path.splitext(filename)[1].lower()
                if file_extension in extensions:
                    files.append(os.path.join(dirpath, filename))
            except Exception as e:
                print(f"Error processing file '{filename}': {e}")
    if not files:
        print(f"No supported audio files ({', '.join(extensions)}) found in '{directory}'.")
    else:
        print(f"Found {len(files)} audio files.")
    return files



def main(args):
    total_start_time = time.time()

    hash_algorithm = hashlib.sha1 if args.hash_algorithm == "sha1" else hashlib.sha256

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Directory '{args.data_dir}' does not exist.")

    fingerprint_generator = FingerprintGenerator(
        sr=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        neighborhood_size=args.peak_neighborhood_size,
        min_amplitude=args.min_amplitude,
        target_t_min=args.target_t_min,
        target_t_max=args.target_t_max,
        target_f_max_delta=args.target_f_max_delta,
        hash_algorithm=hash_algorithm
    )
    
    redis_db = RedisDB()
    # redis_db.clear_db()

    audio_files = get_files(args.data_dir, args.extensions)
    for audio_file in audio_files:
        register_audio(
            audio_file,
            audio_info={"path": audio_file, "filename": os.path.basename(audio_file)},
            fingerprint_generator=fingerprint_generator,
            redis_db=redis_db
        )

    print(f'Took {time.time() - total_start_time:.2f}s to process {len(audio_files)} audio files.')
    
    

    match_id, message = find_match(args.query_file, fingerprint_generator, redis_db)
    print(message)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio fingerprinting and matching")
    parser.add_argument("--sample-rate", type=int, default=22050,
                        help="Sample rate for audio processing (Hz)")
    parser.add_argument("--n-fft", type=int, default=2048,
                        help="Window size for FFT")
    parser.add_argument("--hop-length", type=int, default=512,
                        help="Number of samples between successive frames")
    parser.add_argument("--peak-neighborhood-size", type=int, default=20,
                        help="Neighborhood size for local peak detection (frames x freq_bins)")
    parser.add_argument("--min-amplitude", type=float, default=10,
                        help="Minimum magnitude threshold for peaks")
    parser.add_argument("--target-t-min", type=int, default=5,
                        help="Minimum time delta (frames) between anchor and target")
    parser.add_argument("--target-t-max", type=int, default=100,
                        help="Maximum time delta (frames) between anchor and target")
    parser.add_argument("--target-f-max-delta", type=int, default=100,
                        help="Maximum frequency delta (bins) between anchor and target")
    parser.add_argument("--hash-algorithm", default="sha1", choices=["sha1", "sha256"],
                        help="Hash algorithm to use for fingerprinting")
    parser.add_argument("--data-dir", default="data/",
                        help="Directory to scan recursively for audio files")
    parser.add_argument("--query-file", default="data/fma/music-fma-0004.wav",
                        help="Query audio file path")
    parser.add_argument("--extensions", default=".wav,.mp3,.flac,.ogg,.m4a",
                        help="Comma-separated list of supported audio extensions")

    args = parser.parse_args()
    args.extensions = set(ext if ext.startswith(".") else f".{ext}" for ext in args.extensions.split(","))
    main(args)