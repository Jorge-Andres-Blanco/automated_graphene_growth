import csv
from pathlib import Path

def log_model_decision(filepath: str | Path, frame_index: int, pred_flow: float):
    """
    Fastly appends the model's decision to a CSV file.
    Creates the file and writes a header if it doesn't exist yet.
    """
    path = Path(filepath)
    file_exists = path.is_file()
    
    # Using 'a' mode opens the file for appending without overwriting
    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header only once when the file is first created
        if not file_exists:
            writer.writerow(["frame_index", "pred_optimal_flow"])
            
        # Quickly write the new row
        writer.writerow([frame_index, pred_flow])