from pathlib import Path
import torch
import numpy as np
from data_processing.data_loader import load_transition_data
import WM_JABV.train_transition_model as ttm
import WM_JABV.evaluation as eval
from WM_JABV.transition_models import *
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--hiddim', type=int, required=True)
    parser.add_argument('--activation', type=str, default='relu')
    args = parser.parse_args()
    activation = args.activation
    hidden_dimension = args.hiddim
    hist = 1



    # To be changed according to the executing machine
    train_data_path = Path("/data/lmcat/Computer_vision/training_data")
    validation_data_path = Path("/data/lmcat/Computer_vision/validation_data")

    step_size_list = [4,15,30]
    normalization_list = ["batch", "layer", "instance"]
    sequence_indices = range(0,4)
    steps_ahead = 5

    
    for step_size in step_size_list:
        for normalization in normalization_list:

            ensemble_model = EnsembleTransitionModel(num_models=5,
                                                     latent_dim=384,
                                                     action_dim=1,
                                                     hidden_dim=hidden_dimension,
                                                     normalization=normalization,
                                                     activation=activation,
                                                     num_hidden_layers=2,
                                                     history=hist)
            model_name_prefix = f"/data/lmcat/Computer_vision/models/mlp_activation_{activation}_norm_{normalization}_hist{hist}_step{step_size}_hiddim{hidden_dimension}"

            # Training/calling the model
            try:
                for i, model in enumerate(ensemble_model.models):
                    model.load_state_dict(torch.load(f"{model_name_prefix}_transition_model_{i}.pth"))
            
            except FileNotFoundError:
                
                print(f"Model {i} not found. Training from scratch...")
                ensemble_model, losses = ttm.train_ensmble_with_bagging(ensemble_model=ensemble_model,
                                                                        data_path = train_data_path,
                                                                        save_prefix = model_name_prefix,
                                                                        epochs=4, lr=1e-3, batch_size=64,
                                                                        step_size = step_size)
                
                losses_mean = np.mean(losses, axis=0)
                losses_std = np.std(losses, axis=0)
                print(f"Ensemble training completed. Last loss mean and std: {losses_mean[-1]}, {losses_std[-1]}")


            # Evaluation
            z_eval, a_eval, y_eval, indices = load_transition_data(validation_data_path, step_size=step_size, hist_length=hist, return_indices=True)

            for seq_i in sequence_indices:

                (start_idx, stop_idx) = indices[seq_i] # Target the specific movie of the validation data
                
                # Extract only one sliding window (last available state in that trajectory)
                # Shape becomes (hist, 384) instead of (batch, hist, 384)
                z_hist_np = z_eval[start_idx:stop_idx]
                a_hist_np = a_eval[start_idx:stop_idx]
                
                # 2. Convert to PyTorch Tensors and send to the correct device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                z_hist_tensor = torch.tensor(z_hist_np, dtype=torch.float32, device=device)
                a_hist_tensor = torch.tensor(a_hist_np, dtype=torch.float32, device=device)
                
                # Ensure the model is ready for evaluation
                ensemble_model.to(device)
                ensemble_model.eval()
                    

                plot_path = f"/data/lmcat/Computer_vision/plots/sequence{seq_i}_hist{hist}_step{step_size}_hiddim{hidden_dimension}_norm_{normalization}_activation_{activation}.png"

                eval.plot_actions_vs_time_for_sequence(ensemble_model, z_hist_tensor, a_hist_tensor, history=hist, step_size=step_size, a_pos="all", future_steps=steps_ahead, save_path=plot_path)

    return None


if __name__ == "__main__":

    main()