import argparse
import hashlib
import os
import time

from fingerprint_generator import FingerprintGenerator
from sqlite_db import SQLiteDB


def register_audio(file_path, audio_info, fingerprint_generator, sqlite_db):
    """Registers a single audio file if not already present."""
    
    if sqlite_db.file_already_registered(file_path):
        print(f"Skipping registration: '{os.path.basename(file_path)}' already in database.")
        return False 

    print(f"Registering: '{os.path.basename(file_path)}'")
    start_time = time.time()
    fingerprints = fingerprint_generator.generate_fingerprints(file_path)

    if not fingerprints:
        print(f"Could not generate fingerprints for {file_path}. Skipping registration.")
        return False

    audio_id = sqlite_db.register_audio(file_path, audio_info, fingerprints)

    end_time = time.time()
    if audio_id is not None:
        print(f"Registered '{os.path.basename(file_path)}' (ID: {audio_id}) with {len(fingerprints)} fingerprints. Took {end_time - start_time:.2f}s.")
        return True 
    else:
        print(f"Failed to register '{os.path.basename(file_path)}'. Took {end_time - start_time:.2f}s.")
        return False


def find_match(query_file_path, fingerprint_generator, sqlite_db):
    """Processes a query audio file and attempts to find the best match using SQLite."""
    
    print(f"\n--- Querying File: {os.path.basename(query_file_path)} ---")
    if not os.path.exists(query_file_path):
        return None, f"Query file not found: {query_file_path}"

    start_time = time.time()

    query_fingerprints = fingerprint_generator.generate_fingerprints(query_file_path)

    if not query_fingerprints:
        return None, "No fingerprints generated for query file."

    print(f"Generated {len(query_fingerprints)} fingerprints for query.")

    best_match_audio_id, result_message = sqlite_db.find_match(query_fingerprints)

    match_info_str = ""
    if best_match_audio_id is not None:
        audio_info = sqlite_db.get_audio_info(best_match_audio_id)
        if audio_info:
            match_info_str = f"\n---> Match Info: Filename='{audio_info.get('filename', 'Unknown')}', Path='{audio_info.get('path', 'Unknown')}'"
        else:
            match_info_str = "\n---> Could not retrieve info for matched audio ID."

    end_time = time.time()
    full_message = f"{result_message}{match_info_str}\n--- Search took {end_time - start_time:.2f}s ---"
    return best_match_audio_id, full_message


def get_files(directory, extensions):
    """Recursively retrieves files in a directory matching the supported audio extensions."""
    
    print(f"Scanning directory '{directory}' for audio files...")
    files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            try:
                file_extension = os.path.splitext(filename)[1].lower()
                if file_extension in extensions:
                    full_path = os.path.join(dirpath, filename)
                    if os.access(full_path, os.R_OK):
                        files.append(full_path)
                    else:
                         print(f"Warning: Cannot read file '{full_path}', skipping.")
            except Exception as e:
                print(f"Error processing file '{filename}': {e}")

    if not files:
        print(f"No supported audio files ({', '.join(extensions)}) found or readable in '{directory}'.")
    else:
        print(f"Found {len(files)} supported audio files in '{directory}'.")
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

    sqlite_db = SQLiteDB(db_path=args.db_path)

    if args.clear_db:
        sqlite_db.clear_db()

    print("\n--- Starting Audio Registration ---")
    audio_files = get_files(args.data_dir, args.extensions)
    registered_count = 0
    failed_count = 0
    if audio_files:
        for audio_file in audio_files:
            success = register_audio(
                audio_file,
                audio_info={"path": audio_file, "filename": os.path.basename(audio_file)},
                fingerprint_generator=fingerprint_generator,
                sqlite_db=sqlite_db 
            )
            if success is True:
                 registered_count += 1
            elif success is False:
                 failed_count +=1

        print(f"\n--- Registration Summary ---")
        print(f"Processed {len(audio_files)} files.")
        print(f'Total registration time: {time.time() - total_start_time:.2f}s')
    else:
        print("No audio files found to register.")


    if args.query_file:
        if not os.path.exists(args.query_file):
             print(f"\nError: Query file '{args.query_file}' not found.")
        else:
            match_id, message = find_match(args.query_file, fingerprint_generator, sqlite_db)
            print(message) 
    else:
        print("\nNo query file specified. Skipping matching phase.")


    sqlite_db.close()

    total_end_time = time.time()
    print(f"\n--- Total execution time: {total_end_time - total_start_time:.2f}s ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio fingerprinting and matching")

    parser.add_argument("--db-path", default="fingerprints.db",
                        help="Path to the SQLite database file")
    parser.add_argument("--clear-db", action='store_true',
                        help="Clear the entire database before starting registration")

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
                        help="Directory to scan recursively for audio files to register")
    parser.add_argument("--query-file", default="data/fma/music-fma-0004.wav",
                        help="Query audio file path")
    parser.add_argument("--extensions", default=".wav,.mp3,.flac,.ogg,.m4a",
                        help="Comma-separated list of supported audio extensions")

    args = parser.parse_args()
    
    # Ensures extensions are stored as a set of lowercase strings starting with '.'
    args.extensions = set(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.extensions.split(","))

    main(args)