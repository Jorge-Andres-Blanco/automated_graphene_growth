from pathlib import Path
import h5py
import hdf5plugin
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
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
    """
    Streams images directly from an HDF5 file on disk to the GPU via the DINO encoder.
    """

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
    hidden_dimension=1024
    normalization="layer"
    activation="leaky_relu"
    ensemble_model = ensemble_model = EnsembleTransitionModel(num_models=5,
                                                        latent_dim=384,
                                                        action_dim=1,
                                                        hidden_dim=1024,
                                                        normalization=normalization,
                                                        activation=activation,
                                                        num_hidden_layers=2,
                                                        history=hist)

    model_name_prefix = f"/data/lmcat/Computer_vision/models/mlp_activation_{activation}_norm_{normalization}_hist{hist}_step{step_size}_hiddim{hidden_dimension}"

    # Training/calling the model
    try:
        for i, model in enumerate(ensemble_model.models):
            model.load_state_dict(torch.load(f"{model_name_prefix}_transition_model_{i}.pth", map_location=device))
            print(f"model {i} loaded")
    except FileNotFoundError:
            print(f"Model {i} not found")

    steps_in_future = 0
    movie_num = 2
    initial_frame_idx = 2000

    target_frame_idx = initial_frame_idx + step_size*steps_in_future
    frame_0 = get_frame_data(movie_num, initial_frame_idx)
    frame_1 = get_frame_data(movie_num, target_frame_idx)
    a0 = get_frame_data(movie_num,initial_frame_idx, measurement="CH4")

    # Fetch the SEQUENCE of CH4 flow for the bottom-right plot
    movie_path, file_name, scan_number = DATA_FILES[movie_num]
    full_file_path = os.path.join(movie_path, file_name)
    with h5py.File(full_file_path, "r") as f:
        # Grab the slice from initial to target
        ch4_dataset = f"{scan_number}.1/measurement/CH4"
        actual_flow_sequence = f[ch4_dataset][initial_frame_idx : target_frame_idx]

    # Encoding
    encoder = DinoEncoder()
    embeddings = encode_frames([frame_0, frame_1], encoder=encoder)
    z0, z1 = embeddings[0], embeddings[1]

    l2_distance = np.linalg.norm(z0-z1)
    cosine_similarity = np.dot(z0, z1) / (np.linalg.norm(z0) * np.linalg.norm(z1))
    compare_images_in_latent_space(frame_0, frame_1, z0, z1)


    print(f"L2 Distance: {l2_distance}")
    print(f"Cosine Similarity: {cosine_similarity}")

    losses, actions_evaluated = ensemble_model.predict_action_losses(
        steps=5, 
        z_init=torch.tensor([z0], dtype=torch.float32), 
        a_init=torch.tensor([a0], dtype=torch.float32), 
        a_pos="all", 
        target=torch.tensor(z1, dtype=torch.float32)
    )


    # PLot

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # [ROW 0, COL 0]: Current Image
    axes[0, 0].imshow(frame_0, cmap='gray')
    axes[0, 0].set_title(f"Current State (Frame {initial_frame_idx})", fontsize=14)
    axes[0, 0].axis('off')

    # [ROW 0, COL 1]: Target Image
    axes[0, 1].imshow(frame_1, cmap='gray')
    axes[0, 1].set_title(f"Target State (Frame {target_frame_idx})", fontsize=14)
    axes[0, 1].axis('off')

    # Row 1, col 1: 
    eval.plot_possible_actions_losses(losses, actions_evaluated, aggregate='mean', ax=axes[1, 0])

# [ROW 1, COL 1]: Actual Flow vs Frame
    frames_range = np.arange(initial_frame_idx, target_frame_idx)
    axes[1, 1].plot(frames_range, actual_flow_sequence, marker='o', color='orange', linewidth=2)
    axes[1, 1].set_title("Actual Applied CH4 Flow", fontsize=14)
    axes[1, 1].set_xlabel("Frame Index", fontsize=12)
    axes[1, 1].set_ylabel("CH4 Flow (sccm)", fontsize=12)
    #axes[1, 1].set_ylim(0, np.max(actual_flow_sequence))
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)

    # Add the metrics to the overarching Suptitle
    title_text = (
        f"Transition Analysis\n"
        f"L2 Distance: {l2_distance:.4f}  |  Cosine Similarity: {cosine_similarity:.4f}"
    )
    plt.suptitle(title_text, fontsize=18, fontweight='bold')

    # Polish and display
    plt.tight_layout()
    plt.savefig("Transition_analysis.png", dpi=300) # Save the combined image
    plt.show()

    return None


if __name__ == "__main__":

    main()