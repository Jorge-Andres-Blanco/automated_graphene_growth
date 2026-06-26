import torch
from pathlib import Path
import numpy as np
import time
from src.environment import ReactorEnv
from src.models import DinoEncoder, EnsembleTransitionModel
from src.controllers import CEMPlanner
from src.data_handling import HDF5Processor
from src.utils.logger import log_model_decision
from scripts.functions_online_testing import hold_equilibrium_loop, growth_loop_with_target

LOG_FILE_FOLDER = Path("/data/lmcat/Computer_vision/automated_graphene_growth/logs/")

def main():
    # --- Setup ---
    print("Booting up Autonomous Graphene Control System...")
    log_file = f"autonomous_growth_log_{time.strftime('%Y%m%d-%H%M')}.csv"
    log_file_path = LOG_FILE_FOLDER / log_file
    env = ReactorEnv()
    encoder = DinoEncoder()
    data_processor = HDF5Processor(encoder=encoder)
    
    # Load the trained model
    # We first test the 45-step_size model, then the 30s, If we have time, we also try for the one with 60 
    activation, normalization, hist, step_size, hidden_dimension = "leaky_relu", "layer", 1, 45, 1024

    transition_model = EnsembleTransitionModel(
        num_models=5,
        latent_dim=384,
        action_dim=1,
        hidden_dim=hidden_dimension,
        normalization=normalization,
        activation=activation,
        history=hist,
        num_hidden_layers=2
    ) 
    model_name_prefix = f"/data/lmcat/Computer_vision/models/mlp_activation_{activation}_norm_{normalization}_hist{hist}_step{step_size}_hiddim{hidden_dimension}"

    transition_model.load_ensemble(model_name_prefix)

    # Initialize the brain
    planner = CEMPlanner(transition_model=transition_model, horizon=5)
    
    # Define your target
    movie_num = 7
    initial_frame_idx = 320
    target_frame = data_processor.get_frame_data(movie_num, initial_frame_idx)

    target_image = target_frame
    target_z = data_processor.encode_frames([target_frame])[0]
    target_is_etching=True if target_frame_idx==0 else False
    # New log_file for new target
    log_file = f"autonomous_growth_log_{time.strftime('%Y%m%d-%H%M')}.csv"
    log_file_path = LOG_FILE_FOLDER / log_file
    growth_loop_with_target(env, encoder, planner, target_is_etching, target_z, log_file_path)


    movie_num = 7
    target_frame_idx = 330 # 0 etching, 100 small flakes, 200 medium size flakes with nucleus, 320 bigger flakes without nucleus
    target_frame = data_processor.get_frame_data(movie_num, target_frame_idx)
    target_z = data_processor.encode_frames([target_frame])[0]
    target_is_etching=True if target_frame_idx==0 else False

    # New log_file for new target
    log_file = f"autonomous_growth_log_{time.strftime('%Y%m%d-%H%M')}.csv"
    log_file_path = LOG_FILE_FOLDER / log_file
    growth_loop_with_target(env, encoder, planner, target_is_etching, target_z, log_file_path)




    print("Autonomous growth loop has ended. Please check the log file for details and review")
    hold_equilibrium_loop(env, encoder, planner)


    print("Autonomous growth loop has ended. Please check the log file for details and review")

if __name__ == "__main__":
    main()