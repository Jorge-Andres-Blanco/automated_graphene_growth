from pathlib import Path
import torch
import numpy as np
from src.data_handling import TransitionDataLoader
from src.utils.evaluation import Evaluator
from src.models import EnsembleTransitionModel, Trainer
import gc



def main():

    # To be changed according to the executing machine
    train_data_path = Path("/data/lmcat/Computer_vision/training_data")
    validation_data_path = Path("/data/lmcat/Compuer_vision/validation_data")


    hist_range = [1, 2, 5, 10, 20]
    step_size_range = [2,4,5,6]
    train = True


    for hist in hist_range:

        for step_size in step_size_range:

            run_id = f"hist_{hist}_step_{step_size}"
            print(f"Running evaluation for {run_id}")

            ensemble_model = EnsembleTransitionModel(num_models=5, latent_dim=384, action_dim=1, hidden_dim=512, num_hidden_layers=2, history=hist)
            trainer = Trainer(lr=1e-3, batch_size=64, epochs=5)
            train_data_loader = TransitionDataLoader(train_data_path, step_size=step_size, hist_length=hist)
            validation_data_loader = TransitionDataLoader(validation_data_path, step_size=step_size, hist_length=hist)
            evaluator = Evaluator()
            
   
            # Training 
            
            if train:
                
                ensemble_model = trainer.train_ensmble_with_bagging(ensemble_model=ensemble_model, data_loader=train_data_loader, save_prefix = "bagging")
                losses = trainer.losses
                losses_mean = np.mean(losses, axis=0)
                losses_std = np.std(losses, axis=0)

                trainer.plot_training_loss_vs_epoch(losses_mean)
                print(f"Ensemble training completed. Last loss mean and std: {losses_mean[-1]}, {losses_std[-1]}")
            
            else:
            
                for i, model in enumerate(ensemble_model.models):
                    model.load_state_dict(torch.load(f"{run_id}_transition_model_{i}.pth"))


            
            z_eval, a_eval, y_eval, indices = validation_data_loader.load_full_dataset(return_indices=True)

            l2_distances, cos_similarities, mse_loss = evaluator.evaluate_ensemble_transition_model(ensemble_model, z_eval, a_eval, y_eval)

            print(f"MSE Loss on validation data: {mse_loss}")

            print(indices)

            
            (i, f) = indices[-1] # Just evaluate on the last trajectory

            dz, std_z, l2_distances, cos_similarities = evaluator.evaluate_rollouts(steps = 5, model = ensemble_model, z_traj= z_eval[i:f], a_traj = a_eval[i:f], y_traj = y_eval[i:f])

            # Make into single array and save as .npy file

            save_filename = f"rollout_metrics_{run_id}.npz"
            np.savez(
                save_filename, 
                dz=dz, 
                std_z=std_z, 
                l2_distances=l2_distances, 
                cos_similarities=cos_similarities
            )
            print(f"Saved evaluation metrics to {save_filename}\n")

            # --- MEMORY CLEANUP BLOCK ---
            
            # 1. Delete PyTorch model
            del ensemble_model
            
            # 2. Delete massive RAM arrays
            del z_eval, a_eval, y_eval, dz, std_z, l2_distances, cos_similarities
            
            # 3. Force Python garbage collection
            gc.collect()
            
            # 4. Force PyTorch to release unreferenced VRAM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    return None


if "__main__" == __name__:
    main()