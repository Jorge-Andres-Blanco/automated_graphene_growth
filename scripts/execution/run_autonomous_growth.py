import torch
from pathlib import Path
import numpy as np
import time
from src.environment import ReactorEnv
from src.models import DinoEncoder
from src.controllers import CEMPlanner
from src.data_handling import HDF5Processor
from automated_graphene_growth.scripts.execution.functions_online_testing import hold_equilibrium_loop, growth_loop_with_target
from src.utils.misc import load_model_from_yaml_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
config_path = PROJECT_ROOT / "config.yaml"

def main():
    # --- Setup ---
    print("Booting up Autonomous Graphene Control System...")
    log_file = f"autonomous_growth_log_{time.strftime('%Y%m%d-%H%M')}.csv"
    log_file_path = PROJECT_ROOT / "logs" / log_file
    env = ReactorEnv()
    encoder = DinoEncoder()
    data_processor = HDF5Processor(encoder=encoder)
    
    # Load the trained model
    transition_model = load_model_from_yaml_config(config_path)

    # Initialize the brain
    planner = CEMPlanner(transition_model=transition_model, horizon=5)
    
    # Define your target
    movie_num = 7
    initial_frame_idx = 320
    target_frame = data_processor.get_frame_data(movie_num, initial_frame_idx)

    target_z = data_processor.encode_frames([target_frame])[0]
    target_is_etching=True if target_frame_idx==0 else False
    # New log_file for new target
    log_file = f"autonomous_growth_log_{time.strftime('%Y%m%d-%H%M')}.csv"
    log_file_path = PROJECT_ROOT / "logs" / log_file
    growth_loop_with_target(env, encoder, planner, target_is_etching, target_z, log_file_path)


    movie_num = 7
    target_frame_idx = 330 # 0 etching, 100 small flakes, 200 medium size flakes with nucleus, 320 bigger flakes without nucleus
    target_frame = data_processor.get_frame_data(movie_num, target_frame_idx)
    target_z = data_processor.encode_frames([target_frame])[0]
    target_is_etching=True if target_frame_idx==0 else False

    # New log_file for new target
    log_file = f"autonomous_growth_log_{time.strftime('%Y%m%d-%H%M')}.csv"
    log_file_path = PROJECT_ROOT / "logs" / log_file
    growth_loop_with_target(env, encoder, planner, target_is_etching, target_z, log_file_path)




    print("Autonomous growth loop has ended. Please check the log file for details and review")
    hold_equilibrium_loop(env, encoder, planner)


    print("Autonomous growth loop has ended. Please check the log file for details and review")

if __name__ == "__main__":
    main()