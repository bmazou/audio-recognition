def mmss_to_seconds(mmss):
    """Converts a time string in mm:ss format to seconds."""
    try:
        parts = mmss.strip().split(":")
        if len(parts) != 2:
            print(f"Invalid time format: '{mmss}'. Expected format is mm:ss.")
            return None
        
        
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
        
    except Exception as e:
        print(f"Error converting time '{mmss}': {e}")
        return None