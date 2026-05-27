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
    log_file = f"hold_equilibrium_log_{time.strftime('%Y%m%d-%H%M')}.csv"
    log_file_path = LOG_FILE_FOLDER / log_file
    env = ReactorEnv()
    encoder = DinoEncoder()
    data_processor = HDF5Processor(encoder=encoder)
    
    # Load the trained model
    activation, normalization, hist, step_size, hidden_dimension = "leaky_relu", "layer", 1, 30, 1024

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
    planner = CEMPlanner(transition_model=transition_model, horizon=10)
    
    # Define your target
    print("Observing current state from the reactor to read target state...")
    state = env.observe()
    target_image = state['Image']
    target_z = encoder.encode_numpy_array(target_image)[0]

    # --- The Control Loop ---
    print("Target collected. Starting equilibrium hold loop...")
    predictions = []
    steps = 60
    for step in range(steps): #This should take a bit more than 30 minutes
        
        # Sense the world
        print("Observing current state from the reactor...")
        state = env.observe()
        current_image = state['Image']
        current_flow = state['CH4'][-1]
        
        # Encode to latent space
        current_z = encoder.encode_numpy_array(current_image)[0]

        
        l2_distance = np.linalg.norm(current_z - target_z)
        cosine_similarity = np.dot(current_z, target_z) / (np.linalg.norm(current_z) * np.linalg.norm(target_z))
        
        print(f"Current Metrics -> L2: {l2_distance:.3f} | Cosine: {cosine_similarity:.3f}")

        # Plan
        print("Planning next action using the planner...")
        best_ch4_flow = planner.get_best_action(current_z, current_flow, target_z, action_space="closer_7")
        
        # Write to log
        log_model_decision(filepath=log_file_path, frame_index=env.observer.index, pred_flow=best_ch4_flow)

        # Take action
        print(f"Applying action: Setting CH4 flow to {best_ch4_flow:.2f}")    
        state = env.act(ch4_action=best_ch4_flow)

        print("Sleeping 30 seconds before next observation...")
        time.sleep(30)

    print("Equilibrium hold loop completed after 30 minutes.")
    print(f"Final L2 Distance: {l2_distance:.2f} | Final Cosine Similarity: {cosine_similarity:.2f}")
    print("Autonomous control loop has ended. Please check the log file for details and review")

if __name__ == "__main__":
    main()