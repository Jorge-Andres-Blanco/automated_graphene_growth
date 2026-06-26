import pandas as pd
import numpy as np
import imageio.v3 as iio
import torch
from pathlib import Path
from src.utils.evaluation import Evaluator
from src.utils.misc import compile_video_from_frames
from src.data_handling.hdf5_processor import HDF5Processor
from src.models import DinoEncoder, EnsembleTransitionModel
from src.utils.evaluation import Evaluator
import argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate a video replay of the autonomous growth process from log files.")
    parser.add_argument('--frame_rate', type=int, default=3, help="Frame rate for the output video.")
    args = parser.parse_args()
    # Standard setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_processor = HDF5Processor(encoder=DinoEncoder())
    evaluator = Evaluator(device=device)
    

    hist = 1
    step_size = 30
    hidden_dimension=1024
    normalization="layer"
    activation="leaky_relu"

    # Model Setup

    ensemble_model = EnsembleTransitionModel(num_models=5,
                                            latent_dim=384,
                                            action_dim=1,
                                            hidden_dim=hidden_dimension,
                                            normalization=normalization,
                                            activation=activation,
                                            num_hidden_layers=2,
                                            step_size=step_size,
                                            history=hist)

    model_name_prefix = f"/data/lmcat/Computer_vision/models/mlp_activation_{activation}_norm_{normalization}_hist{hist}_step{step_size}_hiddim{hidden_dimension}"

    # Load weights
    try:
        for i, model in enumerate(ensemble_model.models):
            model.load_state_dict(torch.load(f"{model_name_prefix}_transition_model_{i}.pth", map_location=device))
            print(f"model {i} loaded")
    except FileNotFoundError:
            print(f"Model {i} not found")

    for movie_num in [0,1,2,4]:
        
         # Create video frames from logs
        saved_images, temp_dir = evaluator.generate_video_frames_for_validation(
            movie_num=movie_num,
            model=ensemble_model,
            data_processor=data_processor,
            horizon=3
        )
        
        output_video_path = f"/data/lmcat/Computer_vision/automated_graphene_growth/videos/model_{step_size}/validation_{movie_num}_replay.mp4"        

        # Compile video
        compile_video_from_frames(
            saved_images=None,
            temp_dir=temp_dir,
            output_video_path=output_video_path,
            fps=args.frame_rate
        )