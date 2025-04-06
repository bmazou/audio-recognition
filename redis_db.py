import json
from collections import defaultdict

import redis

"""
SCHEMA:
- audio_filepath:{file_path} -> audio_id
  - Serves to check if the file has already been registered.
  
- audio_info:{audio_id} -> {'filename': '...', 'path': '...'}
  - Stores metadata about the audio_id

- fp:{hash_hex} -> {anchor_time:audio_id}
  - i.e. fingerprint hash -> (anchor_time, audio_id) tuple,
    where anchor_time is the absolute time of the anchor in the audio file
"""

class RedisDB:
    def __init__(self, host='localhost', port=6380, db=0):
        self.client = redis.StrictRedis(host=host, port=port, db=db, decode_responses=True)
    
    
    def clear_db(self):
        """Removes all data from the Redis database."""
        self.client.flushdb()
        print(f"Database has {self.client.dbsize()} entries.")
        
        
    def file_already_registered(self, file_path):
        """Checks if the audio file has already been registered in the database."""
        return self.client.exists(f"audio_filepath:{file_path}") != 0
        
    
    def register_audio(self, file_path, audio_info, fingerprints, audio_id):
        """Saves the audio file information and fingerprints in the database."""
        
        self.client.set(f"audio_filepath:{file_path}", audio_id)
        self.client.set(f"audio_info:{audio_id}", json.dumps(audio_info))
        
        # For each fingerprint, add a record using a set (to avoid duplicates)
        for hash_hex, anchor_time in fingerprints:
            key = f"fp:{hash_hex}"
            value = f'{anchor_time}:{audio_id}'
            self.client.sadd(key, value)


    def _add_potential_matches(self, query_hash, query_anchor_time, potential_matches):
        """
        Retrieves potential matches for a given query hash and adds them to the potential_matches dictionary.
        Potential match means that the query and a db_record have the same hash
          - This means that they have the same (anchor_freq, target_freq, time_delta)
        """

        key = f"fp:{query_hash}"
        if not self.client.exists(key):
            return 
        
        # Retrieve all fingerprint records from the set.
        records = self.client.smembers(key)
        for rec in records:
            # Each record is stored as "db_anchor_time:db_audio_id".
            db_anchor_time_str, db_audio_id_str = rec.split(":")
            db_anchor_time = int(db_anchor_time_str)
            db_audio_id = int(db_audio_id_str)
            potential_matches[db_audio_id].append((db_anchor_time, query_anchor_time))

    def _score_potential_matches(self, potential_matches):
        """
        Scores potential matches based on time-difference alignment.
        For each audio_id, it computes the score based on the number of aligned time differences.
        If a potential match has consistent time differences, it is very likely a match.
        """
        
        match_scores = defaultdict(lambda: defaultdict(int))
        final_scores = {}
        for audio_id, time_pairs in potential_matches.items():
            for db_time, query_time in time_pairs:
                delta = db_time - query_time
                match_scores[audio_id][delta] += 1
                
                
            if match_scores[audio_id]:      # Some matches were found for audio_id 
                best_delta = max(match_scores[audio_id], key=match_scores[audio_id].get)
                final_scores[audio_id] = match_scores[audio_id][best_delta]
            else:
                final_scores[audio_id] = 0
                
        return final_scores
    

    def find_match(self, query_fingerprints):
        """
        Processes a list of query fingerprints and returns the best matching audio file.
        """
        
        # `potential_matches` stores audio_id as keys and a list of tuples (db_time, query_time) as values.
        potential_matches = defaultdict(list)
        for query_hash, query_anchor_time in query_fingerprints:
            self._add_potential_matches(query_hash, query_anchor_time, potential_matches)
        
        if not potential_matches:
            return None, f"No matching hashes found in database."
        

        final_scores = self._score_potential_matches(potential_matches)
        if not final_scores:
            return None, "Could not score any matches."
        
        best_match_audio_id = max(final_scores, key=final_scores.get)
        best_score = final_scores[best_match_audio_id]
        return best_match_audio_id, f"Best Match Audio ID: {best_match_audio_id} with score {best_score}."

    def get_audio_info(self, audio_id):
        """Retrieves the audio information stored in the database for a given audio_id."""
        info_json = self.client.get(f"audio_info:{audio_id}")
        if info_json:
            return json.loads(info_json)
        return None