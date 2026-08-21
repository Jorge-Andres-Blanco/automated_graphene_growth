import torch
from pathlib import Path
import numpy as np
import time
from src.environment import ReactorEnv
from src.models import DinoEncoder
from src.controllers import CEMPlanner
from src.utils.logger import log_model_decision
from src.utils.misc import load_model_from_yaml_config, load_yaml_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
config_path = PROJECT_ROOT / "config.yaml"

def main():
    # --- Setup ---
    config = load_yaml_config(config_path)
    step_size = config['execution']['active_model']['step_size']
    print("Booting up Autonomous Graphene Control System...")
    log_file = f"active_learning_log_{time.strftime('%Y%m%d-%H%M')}.csv"
    log_file_path = PROJECT_ROOT / "logs" / log_file
    env = ReactorEnv()
    encoder = DinoEncoder()
    
    # Load the trained model
    transition_model = load_model_from_yaml_config(config_path)

    # Initialize the brain
    horizon = 3  # You can adjust this based on your needs
    wait_steps = 1  # Time to wait after each action, also reaction time
    planner = CEMPlanner(transition_model=transition_model, horizon=horizon)

    # --- The Control Loop ---
    print("Starting learning loop...")
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

        print(f"Sleeping {step_size*wait_steps} seconds (step size)")
        time.sleep(step_size*wait_steps)

    print("Autonomous learning loop has ended. Please check the log file for details and review")

if __name__ == "__main__":
    main()