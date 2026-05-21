from pathlib import Path
import h5py
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from src.data_handling.hdf5_processor import HDF5Processor
from src.models import DinoEncoder, EnsembleTransitionModel
from src.utils.plotting import compare_images_in_latent_space
from src.utils.evaluation import Evaluator



train_data_path = Path("/data/lmcat/Computer_vision/training_data")
validation_data_path = Path("/data/lmcat/Computer_vision/validation_data")


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

    data_processor = HDF5Processor(encoder=DinoEncoder())

    steps_in_future = 0
    movie_num = 2
    initial_frame_idx = 2000

    target_frame_idx = initial_frame_idx + step_size*steps_in_future
    frame_0 = data_processor.get_frame_data(movie_num, initial_frame_idx)
    frame_1 = data_processor.get_frame_data(movie_num, target_frame_idx)
    a0 = data_processor.get_frame_data(movie_num, initial_frame_idx, measurement="CH4")

    # Encoding
    encoder = DinoEncoder()
    embeddings = data_processor.encode_frames([frame_0, frame_1], encoder=encoder)
    actual_flow_sequence = data_processor.get_frame_data(movie_num, np.arange(initial_frame_idx, target_frame_idx+1), measurement="CH4").flatten()
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