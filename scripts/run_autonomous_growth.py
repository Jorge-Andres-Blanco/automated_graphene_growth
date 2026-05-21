import torch
import numpy as np
import time
from src.environment import ReactorEnv
from src.models.dinov2_encoder import DinoEncoder
from src.models.transition import EnsembleTransitionModel
from src.controllers.cem_planner import CEMPlanner

def main():
    # --- A. Setup ---
    print("Booting up Automated Graphene Control System...")
    env = ReactorEnv()
    encoder = DinoEncoder()
    
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
    planner = CEMPlanner(transition_model=transition_model, horizon=5)
    
    # Define your target (e.g., encode an image of perfect graphene)
    target_image = env.observe()['Image'] # Or load from disk
    target_z = encoder.encode_numpy_array([target_image])[0]

    # --- B. The Control Loop ---
    print("Starting growth loop...")
    predictions = []
    steps = 360
    for step in range(steps): #This should take 60 min
        
        # Sense the world
        state = env.observe()
        current_image = state['Image']
        current_flow = state['CH4']
        # Encode to latent space
        current_z = encoder.encode_numpy_array([current_image])[0]
        
        l2_distance = np.linalg.norm(current_z - target_z)
        cosine_similarity = np.dot(current_z, target_z) / (np.linalg.norm(current_z) * np.linalg.norm(target_z))
        
        print(f"Current Metrics -> L2: {l2_distance:.4f} | Cosine: {cosine_similarity:.4f}")
        
        if l2_distance < 3.0 and cosine_similarity > 0.95:
            print(f"Target state reached after {step} steps!")
            print(f"Final L2 Distance: {l2_distance:.2f} | Final Cosine Similarity: {cosine_similarity:.2f}")
            print("Halting the AI control loop to preserve the graphene flake.")
            
            break

        # Plan
        best_ch4_flow = planner.get_best_action(current_z, current_flow, target_z)
        
        
        # Conditions to take action
        if len(predictions) == 0:
            predictions.append(best_ch4_flow)
            increase = best_ch4_flow > current_flow

        elif best_ch4_flow > current_flow and increase:
            predictions.append(best_ch4_flow)

        elif best_ch4_flow < current_flow and not increase:
            predictions.append(best_ch4_flow)

        else:
            predictions = []

        # Action
        if len(predictions) > 5:
            state = env.act(ch4_action=best_ch4_flow)
            predictions = []


        # Reduce the horizon as time passes
        if step > 0 and step % (steps // 5) == 0:
            planner.horizon = max(1,planner.horizon - 1)
        time.sleep(10)

if __name__ == "__main__":
    main()