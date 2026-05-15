from pathlib import Path
import h5py
import hdf5plugin
import torch
import os
import cv2  # <-- Added OpenCV for video generation
import matplotlib.pyplot as plt
import numpy as np

# Adjust these imports according to your project structure
from data_processing.dinov2_encoder import DinoEncoder
from src.utils.plotting import compare_images_in_latent_space
from WM_JABV.transition_models import EnsembleTransitionModel
import WM_JABV.evaluation as eval

# To be changed according to the executing machine
DATA_FILES = [
        (
         "/data/lmcat/inhouse/20260312/ihma818/id10-surf/20260301/RAW_DATA/CV_test_Gr_1_120326_camera/CV_test_Gr_1_120326_camera_0001",
         "CV_test_Gr_1_120326_camera_0001_with_experimental_data.h5",
            1
        ),
        (
         "/data/lmcat/inhouse/20260316/ihma818/id10-surf/20260301/RAW_DATA/CV_test_Gr_2_160326_camera/CV_test_Gr_2_160326_camera_0001",
         "CV_test_Gr_2_160326_camera_0001_with_experimental_data.h5",
            1
        ),
        (
         "/data/lmcat/inhouse/20260316/ihma818/id10-surf/20260301/RAW_DATA/CV_test_Gr_3_170326_camera/CV_test_Gr_3_170326_camera_0001",
         "CV_test_Gr_3_170326_camera_0001.h5",
            1
        ),
        ("/data/lmcat/inhouse/20260316/ihma818/id10-surf/20260301/RAW_DATA/Gr_4_080426_camera/Gr_4_080426_camera_0001/",
         "Gr_4_080426_camera_0001.h5",
            2
        ),
        (
         "/data/lmcat/inhouse/20260316/ihma818/id10-surf/20260301/RAW_DATA/Gr_5_090426_camera/Gr_5_090426_camera_0001/",
         "Gr_5_090426_camera_0001.h5",
            2
        ),
        (
         "/data/lmcat/inhouse/20260316/ihma818/id10-surf/20260301/RAW_DATA/Gr_6_100426_camera/Gr_6_100426_camera_0001/",
         "Gr_6_100426_camera_0001.h5",
            1
        ),
        (
         "/data/lmcat/inhouse/20260428/ihma832/id10-surf/20260401/RAW_DATA/Gr_1_280426_camera/Gr_1_280426_camera_0001/",
         "Gr_1_280426_camera_0001.h5",
            3
        )
    ]

def process_h5_with_dino(file_name: str, scan_number: str, encoder: DinoEncoder, save_file_name: str = None, sleep_time_basler: int = 2):
    with h5py.File(file_name, "r") as f:
        dataset_path = f"{scan_number}.1/measurement/basler"
        if dataset_path not in f:
            raise KeyError(f"Path {dataset_path} not found in {file_name}")
            
        measurements = f[dataset_path]
        print(f"File: {file_name}")
        print(f"Found dataset -> Shape: {measurements.shape}, Type: {measurements.dtype}, Chunks: {measurements.chunks}")
        
        embeddings = encoder.encode_numpy_array(measurements, batch_size=16)

        if save_file_name is not None:
            np.save(file=save_file_name, arr=embeddings)
    return embeddings

def get_frame_from_h5(file_name:str, scan_number:str, frame_number:int, measurement:str):
    with h5py.File(file_name, "r") as f:
        dataset_path = f"{scan_number}.1/measurement/{measurement}"
        if dataset_path not in f:
            raise KeyError(f"Path {dataset_path} not found in {file_name}")
        measurements = f[dataset_path]
        frame = measurements[frame_number]
    return frame

def get_frame_data(movie_num, frame_num, measurement = "basler"):
    movie_path, file_name, scan_number = DATA_FILES[movie_num]
    full_file_path = os.path.join(movie_path, file_name)
    frame = get_frame_from_h5(full_file_path, scan_number, frame_num, measurement)
    return frame

def encode_frames(frames, encoder: DinoEncoder):
    frames = np.stack(frames, axis=0)
    embeddings = encoder.encode_numpy_array(frames, batch_size=16)
    return embeddings

train_data_path = Path(r"\\dfs\data\lmcat\Computer_vision\training_data")
validation_data_path = Path(r"\\dfs\data\lmcat\Computer_vision\validation_data")
    
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hist = 1
    step_size = 30
    hidden_dimension = 1024
    normalization = "layer"
    activation = "leaky_relu"
    
    ensemble_model = EnsembleTransitionModel(num_models=5,
                                             latent_dim=384,
                                             action_dim=1,
                                             hidden_dim=hidden_dimension,
                                             normalization=normalization,
                                             activation=activation,
                                             num_hidden_layers=2,
                                             history=hist)

    model_name_prefix = f"/data/lmcat/Computer_vision/models/mlp_activation_{activation}_norm_{normalization}_hist{hist}_step{step_size}_hiddim{hidden_dimension}"

    # Training/calling the model
    try:
        for i, model in enumerate(ensemble_model.models):
            model.load_state_dict(torch.load(f"{model_name_prefix}_transition_model_{i}.pth", map_location=device))
            # Move model to device just in case
            model.to(device)
            print(f"model {i} loaded")
    except FileNotFoundError:
            print(f"Model {i} not found")

    # ==========================================
    # VIDEO CONFIGURATION
    # ==========================================
    movie_num = 1
    steps_in_future = 5
    start_frame = 200
    end_frame = 350       # Define how many frames you want in the video
    fps = 5.0             # Frames per second for the output video
    
    # Figure size configuration
    fig_width, fig_height = 16, 10
    dpi = 100
    
    # Initialize OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video_path = "Transition_analysis_video.mp4"
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (fig_width * dpi, fig_height * dpi))

    print(f"Starting video generation. Saving to {out_video_path}...")

    # Fetch the ENTIRE sequence of CH4 flow once to speed up the loop
    movie_path, file_name, scan_number = DATA_FILES[movie_num]
    full_file_path = os.path.join(movie_path, file_name)
    with h5py.File(full_file_path, "r") as f:
        ch4_dataset = f"{scan_number}.1/measurement/CH4"
        full_actual_flow = np.array(f[ch4_dataset])

    encoder = DinoEncoder()

    # ==========================================
    # THE RENDERING LOOP
    # ==========================================
    for initial_frame_idx in range(start_frame, end_frame):
        
        target_frame_idx = initial_frame_idx + step_size * steps_in_future
        
        # Stop if we reach the end of the data
        if target_frame_idx >= len(full_actual_flow):
            print("Reached the end of the HDF5 data. Stopping video generation.")
            break

        # 1. Fetch Data
        frame_0 = get_frame_data(movie_num, initial_frame_idx)
        frame_1 = get_frame_data(movie_num, target_frame_idx)
        a0 = get_frame_data(movie_num, initial_frame_idx, measurement="CH4")
        
        # Grab the flow slice from our pre-loaded array
        actual_flow_sequence = full_actual_flow[initial_frame_idx : target_frame_idx]

        # 2. Encoding
        embeddings = encode_frames([frame_0, frame_1], encoder=encoder)
        z0, z1 = embeddings[0], embeddings[1]

        l2_distance = np.linalg.norm(z0 - z1)
        cosine_similarity = np.dot(z0, z1) / (np.linalg.norm(z0) * np.linalg.norm(z1))

        # 3. Model Inference (Ensure inputs are on the right device)
        losses, actions_evaluated = ensemble_model.predict_action_losses(
            steps=5, 
            z_init=torch.tensor([z0], dtype=torch.float32).to(device), 
            a_init=torch.tensor([a0], dtype=torch.float32).to(device), 
            a_pos="all", 
            target=torch.tensor(z1, dtype=torch.float32).to(device)
        )

        # 4. Create Plot
        fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height), dpi=dpi)

        # [ROW 0, COL 0]: Current Image
        axes[0, 0].imshow(frame_0, cmap='gray')
        axes[0, 0].set_title(f"Current State (Frame {initial_frame_idx})", fontsize=14)
        axes[0, 0].axis('off')

        # [ROW 0, COL 1]: Target Image
        axes[0, 1].imshow(frame_1, cmap='gray')
        axes[0, 1].set_title(f"Target State (Frame {target_frame_idx})", fontsize=14)
        axes[0, 1].axis('off')

        # [ROW 1, COL 0]: Action Losses
        eval.plot_possible_actions_losses(losses, actions_evaluated, aggregate='mean', ax=axes[1, 0])

        # [ROW 1, COL 1]: Actual Flow vs Frame
        frames_range = np.arange(initial_frame_idx, target_frame_idx)
        axes[1, 1].plot(frames_range, actual_flow_sequence, marker='o', color='orange', linewidth=2)
        axes[1, 1].set_title("Actual Applied CH4 Flow", fontsize=14)
        axes[1, 1].set_xlabel("Frame Index", fontsize=12)
        axes[1, 1].set_ylabel("CH4 Flow (sccm)", fontsize=12)
        
        # Keep y-axis scale consistent across frames
        axes[1, 1].set_ylim(0, np.max(full_actual_flow) * 1.1) 
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)

        title_text = (
            f"Transition Analysis\n"
            f"L2 Distance: {l2_distance:.4f}  |  Cosine Similarity: {cosine_similarity:.4f}"
        )
        plt.suptitle(title_text, fontsize=18, fontweight='bold')
        plt.tight_layout()

        # 5. Convert Figure to OpenCV Image
        fig.canvas.draw()
        img_buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_rgb = img_buffer.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Matplotlib uses RGB, OpenCV uses BGR
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # 6. Write to Video and Cleanup Memory
        writer.write(img_bgr)
        plt.close(fig) # EXTREMELY IMPORTANT: Prevents RAM from filling up!

        if initial_frame_idx % 10 == 0:
            print(f"Processed frame {initial_frame_idx} / {end_frame}")

    # Finalize Video
    writer.release()
    print(f"Done! Video successfully saved to {out_video_path}")

if __name__ == "__main__":
    main()