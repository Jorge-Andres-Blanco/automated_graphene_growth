import numpy as np
import time
from pathlib import Path
from src.utils.logger import log_model_decision

def hold_equilibrium_loop(env, encoder, planner):
    
    print("Initializing equilibrium Loop")
    LOG_FILE_FOLDER = Path("/data/lmcat/Computer_vision/automated_graphene_growth/logs/")


    log_file = f"hold_equilibrium_log_{time.strftime('%Y%m%d-%H%M')}.csv"
    log_file_path = LOG_FILE_FOLDER / log_file
    
    # Define target
    print("Observing current state from the reactor to read target state...")
    state = env.observe()
    target_image = state['Image']
    target_z = encoder.encode_numpy_array(target_image)[0]
    steps = 720


    for step in range(steps): #This should take a bit more than 60 minutes
        
        # Sense the world
        state = env.observe()
        current_image = state['Image']
        current_flow = state['CH4'][-1]
        
        # Encode to latent space
        current_z = encoder.encode_numpy_array(current_image)[0]

        
        l2_distance = np.linalg.norm(current_z - target_z)
        cosine_similarity = np.dot(current_z, target_z) / (np.linalg.norm(current_z) * np.linalg.norm(target_z))
        
        print(f"Current Metrics -> L2: {l2_distance:.3f} | Cosine: {cosine_similarity:.3f}")

        # Plan
        best_ch4_flow = planner.get_best_action(current_z, current_flow, target_z, action_space="closer_7")
        
        # Write to log
        log_model_decision(filepath=log_file_path, frame_index=env.observer.index, pred_flow=best_ch4_flow)

        # Take action
        print(f"Applying action: Setting CH4 flow to {best_ch4_flow:.2f}")    
        state = env.act(ch4_action=best_ch4_flow)
        time.sleep(5)


def growth_loop_with_target(env, encoder, planner, target_is_etching, target_z, log_file_path, l2_threshold=10):
    # --- The Control Loop ---
    print("Starting growth loop...")
    predictions = []
    steps = 720
    for step in range(steps): #This should take a bit more than 1h
        
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

        l2_threshold = 2 if target_is_etching else l2_threshold
        
        if l2_distance < l2_threshold and cosine_similarity > 0.85:
            print(f"Target state reached after {step} steps!")
            print(f"Final L2 Distance: {l2_distance:.2f} | Final Cosine Similarity: {cosine_similarity:.2f}")
            print("Halting the AI control loop to preserve the graphene flake.")
            
            break

        # Plan
        best_ch4_flow = planner.get_best_action(current_z, current_flow, target_z, action_space="all")
        
        # Write to log
        log_model_decision(filepath=log_file_path, frame_index=env.observer.index, pred_flow=best_ch4_flow)

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
        if len(predictions) >= 3:
            
            mean_last_3_predictions = np.mean(predictions[-3:]) # The action only considers the last ~ 30s

            new_ch4_flow = int(max(mean_last_3_predictions, 0))
            
            print(f"Applying action: Setting CH4 flow to {new_ch4_flow:.2f}")
            
            state = env.act(ch4_action=new_ch4_flow)
            predictions = []


        print("Sleeping 5 seconds...")
        time.sleep(5)