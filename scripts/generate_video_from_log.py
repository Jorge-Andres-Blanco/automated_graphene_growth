import os
import pandas as pd
import numpy as np
import imageio.v3 as iio
import torch
from pathlib import Path
from src.utils.logger import generate_video_frames_from_logs
from src.utils.misc import cleanup_directory, compile_video_from_frames
from src.data_handling.hdf5_processor import HDF5Processor
from src.models import DinoEncoder, EnsembleTransitionModel
from src.utils.evaluation import Evaluator


if __name__ == "__main__":
    
    # Standard setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_processor = HDF5Processor(encoder=DinoEncoder())
    evaluator = Evaluator(device=device)
    
    # Define log and target frame for video generation
    log_name = "autonomous_growth_log_20260527-1709"
    target_frame_movie_num = 7
    target_frame_idx = 320
    target_frame = data_processor.get_frame_data(target_frame_movie_num, target_frame_idx)

    # Model Setup
    
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

    # Load weights
    try:
        for i, model in enumerate(ensemble_model.models):
            model.load_state_dict(torch.load(f"{model_name_prefix}_transition_model_{i}.pth", map_location=device))
            print(f"model {i} loaded")
    except FileNotFoundError:
            print(f"Model {i} not found")
    """    
    # Create video frames from logs
    saved_images, temp_dir = generate_video_frames_from_logs(
        csv_log_path=f"/data/lmcat/Computer_vision/automated_graphene_growth/logs/{log_name}.csv",
        movie_num=9,
        target_frame=target_frame,
        model=ensemble_model,
        data_processor=data_processor,
        evaluator=evaluator
    )
    """

    # Compile video
    compile_video_from_frames(
        saved_images=None,
        temp_dir=Path("/data/lmcat/Computer_vision/automated_graphene_growth/plots/temp_video_frames"),
        output_video_path=f"/data/lmcat/Computer_vision/automated_graphene_growth/videos/{log_name}_replay.mp4",
        fps=3
    )