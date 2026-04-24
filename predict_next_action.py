from pathlib import Path
import torch
import numpy as np
from data_processing.data_loader import load_transition_data
import WM_JABV.train_transition_model as ttm
import WM_JABV.evaluation as eval
from WM_JABV.transition_models import *


def main():

    # To be changed according to the executing machine
    train_data_path = Path(r"\\dfs\data\lmcat\Computer_vision\training_data")
    validation_data_path = Path(r"\\dfs\data\lmcat\Computer_vision\validation_data")


    hist = 1
    step_size = 4
    context_needed = hist*step_size
    train = False
    ensemble_model = EnsembleTransitionModel(num_models=5, latent_dim=384, action_dim=1, hidden_dim=512, num_hidden_layers=2, history=hist)
    model_name_prefix = f"bagging_hist{hist}_step{step_size}"

    # Training 
    if train:
        
        ensemble_model, losses = ttm.train_ensmble_with_bagging(ensemble_model=ensemble_model,
                                                                data_path = train_data_path,
                                                                save_prefix = model_name_prefix,
                                                                epochs=5, lr=1e-3, batch_size=64,
                                                                step_size = step_size)
        
        losses_mean = np.mean(losses, axis=0)
        losses_std = np.std(losses, axis=0)

        ttm.plot_training_loss(losses_mean)
        print(f"Ensemble training completed. Last loss mean and std: {losses_mean[-1]}, {losses_std[-1]}")
    
    else:
    
        for i, model in enumerate(ensemble_model.models):
            model.load_state_dict(torch.load(f"{model_name_prefix}_transition_model_{i}.pth"))


    # Evaluation
    z_eval, a_eval, y_eval, indices = load_transition_data(validation_data_path, step_size=step_size, hist_length=hist, return_indices=True)
    
    (start_idx, stop_idx) = indices[2] # Target the specific movie of the validation data
    
    # Extract only one sliding window (last available state in that trajectory)
    # Shape becomes (hist, 384) instead of (batch, hist, 384)
    z_hist_np = z_eval[start_idx:stop_idx]
    a_hist_np = a_eval[start_idx:stop_idx]
    y_target_np = y_eval[start_idx + (4 * step_size)]
    
    # 2. Convert to PyTorch Tensors and send to the correct device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z_hist_tensor = torch.tensor(z_hist_np, dtype=torch.float32, device=device)
    a_hist_tensor = torch.tensor(a_hist_np, dtype=torch.float32, device=device)
    y_target_tensor = torch.tensor(y_target_np, dtype=torch.float32, device=device)
    
    # Ensure the model is ready for evaluation
    ensemble_model.to(device)
    ensemble_model.eval()
    
    # 3. Run the prediction
    losses, actions_evaluated = ensemble_model.predict_action_losses(
        steps=5, 
        z_init=z_hist_tensor[0], 
        a_init=a_hist_tensor[0], 
        a_pos="all", 
        target=y_target_tensor # Keeps current state
    )
    

    eval.plot_possible_actions_losses(losses, actions_evaluated, aggregate='mean')

    eval.plot_actions_vs_time_for_sequence(ensemble_model, z_hist_tensor, a_hist_tensor, history=hist, step_size=step_size, a_pos="all")
    return None


if __name__ == "__main__":

    main()