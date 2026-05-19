from pathlib import Path
import torch
import numpy as np
from src.data_handling import TransitionDataLoader
from src.utils.evaluation import Evaluator
from src.models import EnsembleTransitionModel, Trainer


def main():

    # To be changed according to the executing machine
    train_data_path = Path(r"\\dfs\data\lmcat\Computer_vision\training_data")
    validation_data_path = Path(r"\\dfs\data\lmcat\Computer_vision\validation_data")
    hist = 15
    step_size = 5
    train = True
    ensemble_model = EnsembleTransitionModel(num_models=5, latent_dim=384, action_dim=1, hidden_dim=512, num_hidden_layers=2, history=hist)
    
    # Training 
    if train:

        trainer = Trainer(lr=1e-3, batch_size=64, epochs=5)
        train_data_loader = TransitionDataLoader(train_data_path, step_size=step_size, hist_length=hist)
        
        ensemble_model = trainer.train_ensmble_with_bagging(ensemble_model=ensemble_model, data_loader=train_data_loader, save_prefix = "bagging")
        losses = trainer.losses
        losses_mean = np.mean(losses, axis=0)
        losses_std = np.std(losses, axis=0)

        trainer.plot_training_loss_vs_epoch(losses_mean)
        print(f"Ensemble training completed. Last loss mean and std: {losses_mean[-1]}, {losses_std[-1]}")
    
    else:
    
        for i, model in enumerate(ensemble_model.models):
            model.load_state_dict(torch.load(f"transition_model_{i}.pth"))


    # Evaluation

    validation_data_loader = TransitionDataLoader(validation_data_path, step_size=step_size, hist_length=hist)
    z_eval, a_eval, y_eval, indices = validation_data_loader.load_full_dataset(return_indices=True)
    
    evaluator = Evaluator()
    l2_distances, cos_similarities, mse_loss = evaluator.evaluate_ensemble_transition_model(ensemble_model, z_eval, a_eval, y_eval)


    print(f"MSE Loss on validation data: {mse_loss}")

    print(indices)

    for (i, f) in indices:
        if (f-i) < hist+1:
            print(f"Skipping evaluation for indices {i} to {f} due to insufficient length.")
            continue
        
        (y_pca, y_pred_pca), l2_distances, cos_similarities = evaluator.evaluate_ensemble_on_trajectory(ensemble_model, z_eval[i:f], a_eval[i:f], y_eval[i:f])
        evaluator.plot_trajectory_evaluation(y_pca, y_pred_pca, l2_distances, cos_similarities)

    return None


if __name__ == "__main__":

    main()