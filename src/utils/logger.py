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


def generate_video_frames_from_logs(csv_log_path: str | Path | None,
                             movie_num: int,
                             target_frame: np.ndarray | None,
                             model: EnsembleTransitionModel,
                             data_processor: HDF5Processor,
                             evaluator: Evaluator) -> tuple[list[str | Path], str | Path]:
    """
    Reads model decisions from a CSV log, generates sequential plots, and compiles them into an MP4.
    """
    # Read the log file
    if csv_log_path is not None:
        df = pd.read_csv(csv_log_path)
        log_indices = df['frame_index'].values.astype(int)
        log_pred_flows = df['pred_optimal_flow'].values
        frames_to_process = len(log_indices)
    else:
        # Validation sequence from manual partitioning (see scripts/manual_data_partitioning.py)
        if movie_num == 0:
            log_indices = np.arange(1800,2200,2)
        elif movie_num == 1:
            log_indices = np.arange(2600,2900,2)
        elif movie_num == 2:
            log_indices = np.arange(1700,2100,2)
        elif movie_num == 4:
            log_indices = np.arange(0,500,2)
        else:
            total_frames = data_processor.get_length_of_measurement_sequence(movie_num)
            log_indices = np.arange(0, total_frames, 5)
                    
        log_pred_flows = None # Dummy values, won't be used
        frames_to_process = len(log_indices) # We will predict 15 frames into the future, so we need to stop 15 frames before the end

    # Prepare temporary folder for frames
    temp_dir = Path("/data/lmcat/Computer_vision/automated_graphene_growth/plots/temp_video_frames")
    temp_dir.mkdir(exist_ok=True)
    saved_images = []

    if target_frame is None:
            target_frame = current_frame_idx + 30 # Target is 2*15 frames ahead


    for i in range(frames_to_process):
        current_frame_idx = log_indices[i]
        if log_pred_flows is None:
            target_idx = current_frame_idx + 30
            target_frame = data_processor.get_frame_data(movie_num, target_idx)

        # Get current frame and action
        frame_0 = data_processor.get_frame_data(movie_num, current_frame_idx)
        a0 = data_processor.get_frame_data(movie_num, current_frame_idx, measurement="CH4")

        # Slicing history up to the current index (inclusive)
        current_history_indices = log_indices[:i+1]
        
        # Note: get_frame_data accepts an array of indices to return sequence data
        actual_flow = data_processor.get_frame_data(movie_num, current_history_indices, measurement="CH4").flatten() #+(2533-2428)
        
        pred_flows = log_pred_flows[:i+1] if log_pred_flows is not None else None

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
            frame_idx=current_frame_idx,
            target_idx=target_idx,
            horizon=4
        )
        saved_images.append(save_path)
        print(f"Rendered frame {i+1}/{frames_to_process}")

    return saved_images, temp_dir