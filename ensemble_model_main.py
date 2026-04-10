from pathlib import Path
import torch
import numpy as np
from data_processing.data_loader import load_transition_data
import WM_JABV.train_transition_model as ttm
import WM_JABV.evaluation as eval
from WM_JABV.transition_models import EnsembleTransitionModel


def main():

    # To be changed according to the executing machine
    train_data_path = Path(r"\\dfs\data\lmcat\Computer_vision\training_data")
    validation_data_path = Path(r"\\dfs\data\lmcat\Computer_vision\validation_data")


    hist = 19
    step_size = 6
    train = True
    ensemble_model = EnsembleTransitionModel(num_models=6, latent_dim=384, action_dim=1, hidden_dim=256, num_hidden_layers=2, history=hist)
    
    if train:
        
        z_train, a_train, y_train = load_transition_data(train_data_path, step_size = step_size, hist_length = hist)
        ensemble_model, losses = ttm.train_ensemble_transition_model(z_train, a_train, y_train, ensemble_model=ensemble_model, epochs=25, lr=2e-3, batch_size=64)
        losses_mean = np.mean(losses, axis=0)
        losses_std = np.std(losses, axis=0)

        ttm.plot_training_loss(losses_mean)
        ttm.plot_training_loss(losses_std)
        print(f"Ensemble training completed. Last loss mean and std: {losses_mean[-1]}, {losses_std[-1]}")
    
    else:
    
        for i, model in enumerate(ensemble_model.models):
            model.load_state_dict(torch.load(f"transition_model_{i}.pth"))


    z_eval, a_eval, y_eval, indices = load_transition_data(validation_data_path, step_size = step_size, hist_length = hist, return_indices=True)
    print (z_eval.shape, a_eval.shape, y_eval.shape)

    l2_distances, cos_similarities, mse_loss = eval.evaluate_ensemble_transition_model(ensemble_model, z_eval, a_eval, y_eval)

    print(f"MSE Loss on validation data: {mse_loss}")

    print(indices)

    for (i, f) in indices:
        if (f-i) < hist+1:
            print(f"Skipping evaluation for indices {i} to {f} due to insufficient length.")
            continue
        (y_pca, y_pred_pca), l2_distances, cos_similarities = eval.evaluate_ensemble_on_trajectory(ensemble_model, z_eval[i:f], a_eval[i:f], y_eval[i:f])
        eval.plot_trajectory_evaluation(y_pca, y_pred_pca, l2_distances, cos_similarities)

    return None


if __name__ == "__main__":

    main()