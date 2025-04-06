import argparse
import hashlib
import os
import time
from collections import defaultdict

from fingerprint_generator import FingerprintGenerator

fingerprint_database = defaultdict(list)
audio_id_to_info = {}


def add_fingerprints_to_db(fingerprints):
    for hash_hex, anchor_time, audio_id in fingerprints:
        fingerprint_database[hash_hex].append((anchor_time, audio_id))

def register_audio(file_path, audio_info, fingerprint_generator):
    
    start_time = time.time()
    fingerprints, audio_id = fingerprint_generator.generate_fingerprints(file_path)
    if not fingerprints:
        return None

    add_fingerprints_to_db(fingerprints)

    if audio_info is None:
        audio_info = {"filename": os.path.basename(file_path), "path": file_path}
    if "path" not in audio_info:
        audio_info["path"] = file_path

    audio_id_to_info[audio_id] = audio_info

    end_time = time.time()
    print(f"OK (ID: {audio_id}, {len(fingerprints)} fps, {end_time - start_time:.2f}s)")
    return audio_id

def find_match(query_file_path, fingerprint_generator):
    """Processes a query audio file and attempts to find the best match in the database."""
    print(f"\nQuerying with: {query_file_path}")
    start_time = time.time()

    # Generate fingerprints for the query audio (using audio_id=-1 for queries)
    query_fingerprints, _ = fingerprint_generator.generate_fingerprints(query_file_path, is_query=True)
    if not query_fingerprints:
        return None, "No fingerprints generated for query"

    print(f"Query: Generated {len(query_fingerprints)} fingerprints.")

    potential_matches = defaultdict(list)
    hashes_matched = 0

    for query_hash, query_anchor_time, _ in query_fingerprints:
        if query_hash in fingerprint_database:
            hashes_matched += 1
            for db_anchor_time, db_audio_id in fingerprint_database[query_hash]:
                potential_matches[db_audio_id].append((db_anchor_time, query_anchor_time))

    if not potential_matches:
        return None, f"No matching hashes found in database. ({time.time() - start_time:.2f}s)"

    print(f"Query: Found {hashes_matched} hash matches corresponding to {len(potential_matches)} potential audio files.")

    match_scores = defaultdict(lambda: defaultdict(int))
    final_scores = {}

    for audio_id, time_pairs in potential_matches.items():
        for db_time, query_time in time_pairs:
            delta = db_time - query_time
            match_scores[audio_id][delta] += 1

        if match_scores[audio_id]:
            best_delta = max(match_scores[audio_id], key=match_scores[audio_id].get)
            final_scores[audio_id] = match_scores[audio_id][best_delta]
        else:
            final_scores[audio_id] = 0

    if not final_scores:
        return None, "Could not score any matches."

    best_match_audio_id = max(final_scores, key=final_scores.get)
    best_score = final_scores[best_match_audio_id]

    end_time = time.time()
    match_info = audio_id_to_info.get(best_match_audio_id, {"filename": "Unknown"})
    
    print('Best 5 matches:')
    for audio_id, score in sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
        info = audio_id_to_info.get(audio_id, {"filename": "Unknown"})
        print(f"  {info['filename']} ({score})")
    
    result_message = f"Best Match: {match_info.get('filename', 'N/A')} with score {best_score}. Search took {end_time - start_time:.2f}s"
    
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

    audio_files = get_files(args.data_dir, args.extensions)
    for audio_file in audio_files:
        register_audio(
            audio_file,
            audio_info={"path": audio_file, "filename": os.path.basename(audio_file)},
            fingerprint_generator=fingerprint_generator
        )

    print(f'Took {time.time() - total_start_time:.2f}s to process {len(audio_files)} audio files.')
    print(f"Database contains fingerprints for {len(audio_id_to_info)} audio files.")
    
    db_hash_count = len(fingerprint_database)
    db_entry_count = sum(len(v) for v in fingerprint_database.values())
    print(f"Database hash index size: {db_hash_count} unique hashes, {db_entry_count} total entries.")

    match_id, message = find_match(args.query_file, fingerprint_generator)
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
    parser.add_argument("--query-file", default="data/fma/music-fma-0002.wav",
                        help="Query audio file path")
    parser.add_argument("--extensions", default=".wav,.mp3,.flac,.ogg,.m4a",
                        help="Comma-separated list of supported audio extensions")

    args = parser.parse_args()
    args.extensions = set(ext if ext.startswith(".") else f".{ext}" for ext in args.extensions.split(","))
    main(args)