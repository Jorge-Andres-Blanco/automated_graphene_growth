import csv
from pathlib import Path
import numpy as np
import pandas as pd
from src.models import EnsembleTransitionModel
from src.data_handling import HDF5Processor
from src.utils.evaluation import Evaluator


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


def generate_video_frames_from_logs(csv_log_path: str | Path,
                             movie_num: int,
                             target_frame: np.ndarray,
                             model: EnsembleTransitionModel,
                             data_processor: HDF5Processor,
                             evaluator: Evaluator) -> tuple[list[str | Path], str | Path]:
    """
    Reads model decisions from a CSV log, generates sequential plots, and compiles them into an MP4.
    """
    # Read the log file
    df = pd.read_csv(csv_log_path)
    log_indices = df['frame_index'].values.astype(int)
    log_pred_flows = df['pred_optimal_flow'].values
    
    # Prepare temporary folder for frames
    temp_dir = Path("/data/lmcat/Computer_vision/automated_graphene_growth/plots/temp_video_frames")
    temp_dir.mkdir(exist_ok=True)
    saved_images = []
    print(f"Generating {len(df)} frames from logs...")
    # Loop through the logs step-by-step
    for i in range(len(df)):
        current_frame_idx = log_indices[i]

        # Get current frame and action
        frame_0 = data_processor.get_frame_data(movie_num, current_frame_idx)
        a0 = data_processor.get_frame_data(movie_num, current_frame_idx, measurement="CH4")

        # Slicing history up to the current index (inclusive)
        current_history_indices = log_indices[:i+1]
        
        # Note: get_frame_data accepts an array of indices to return sequence data
        actual_flow = data_processor.get_frame_data(movie_num, current_history_indices, measurement="CH4").flatten() #+(2533-2428)
        pred_flows = log_pred_flows[:i+1]

        # Plot and save
        save_path = temp_dir / f"frame_{i:04d}.png"
        evaluator.analyze_and_plot_transition(
            model=model,
            data_processor=data_processor,
            frame_0=frame_0,
            frame_1=target_frame,
            a0=a0,
            save_path=save_path,
            frames_range=current_history_indices, # Exact x-axis points
            actual_flow_sequence=actual_flow,
            predicted_flow_sequence=pred_flows,
            frame_idx=current_frame_idx
        )
        saved_images.append(save_path)
        print(f"Rendered frame {i+1}/{len(df)}")

    return saved_images, temp_dir