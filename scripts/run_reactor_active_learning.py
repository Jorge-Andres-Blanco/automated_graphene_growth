import torch
from pathlib import Path
import numpy as np
import time
from src.environment import ReactorEnv
from src.models import DinoEncoder, EnsembleTransitionModel
from src.controllers import CEMPlanner
from src.data_handling import HDF5Processor
from src.utils.logger import log_model_decision

LOG_FILE_FOLDER = Path("/data/lmcat/Computer_vision/automated_graphene_growth/logs/")

def main():
    # --- Setup ---
    print("Booting up Autonomous Graphene Control System...")
    log_file = f"active_learning_log_{time.strftime('%Y%m%d-%H%M')}.csv"
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
    planner = CEMPlanner(transition_model=transition_model, horizon=2)

    # --- The Control Loop ---
    print("Starting growth loop...")
    predictions = []
    steps = 720
    for step in range(steps): #This should take a bit more than 2h
        
        # Sense the world
        print("Observing current state from the reactor...")
        state = env.observe()
        current_image = state['Image']
        current_flow = state['CH4'][-1]
        
        # Encode to latent space
        current_z = encoder.encode_numpy_array(current_image)[0]

        # Plan
        best_ch4_flow = planner.get_highest_variance_action(current_z, current_flow, action_space="all")
        
        # Write to log
        log_model_decision(filepath=log_file_path, frame_index=env.observer.index, pred_flow=best_ch4_flow)

        new_ch4_flow = int(best_ch4_flow)
            
        print(f"Applying action: Setting CH4 flow to {new_ch4_flow:.2f}")
        
        state = env.act(ch4_action=new_ch4_flow)

        print(f"Sleeping {step_size*2} seconds (step size)")
        time.sleep(step_size*2)

    print("Autonomous growth loop has ended. Please check the log file for details and review")

if __name__ == "__main__":
    main()