import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gc
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.data_handling import TransitionDataLoader
    from src.models import TransitionModel, EnsembleTransitionModel

class Trainer:

    def __init__(self, lr: float=1e-3, batch_size: int=64, epochs: int=50):
        """
        Trains the transition models.

        Parameters:
            lr: learning rate
            epochs: times to repeat the training loop
            batch_size: batches in which the training data is divided
        """

        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.losses = None
        
        
        
    def train_transition_model(self, model: 'TransitionModel', z_data: np.ndarray, a_data: np.ndarray, y_data: np.ndarray, save_model_as: str= "transition_model.pth") -> 'TransitionModel':
        
        """
        Trains the transition model on collected data.

        Parameters:
            model to train
            z_data shape (total_samples, history, latent_dim)
            a_data shape (total_samples, history, action_dim)
            y_data shape (total_samples, latent_dim)
            save_model_as model name (.pth), empty to not save

        Returns:
            trained model
            losses array
        """

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")


        model.to(device)
        model.train()

        z_t, a_t, y_t = torch.from_numpy(z_data).float(), torch.from_numpy(a_data).float(), torch.from_numpy(y_data).float()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        n_samples = z_t.shape[0]
        
        losses = []

        mse_loss = nn.MSELoss(reduction='none') # Compute the MSE loss for each sample separately


        for epoch in range(self.epochs):


            # Shuffle
            perm = torch.randperm(n_samples)

            epoch_loss = 0.0
            n_batches = 0       

            for i in range(0, n_samples, self.batch_size):

                optimizer.zero_grad()

                idx = perm[i:i + self.batch_size]            

                # Current batch to device
                # Note: z_batch_i and a_batch_i will have shape (batch_size, history, dim), y_batch_i will have shape (batch_size, dim)
                z_batch_i = z_t[idx].to(device)
                y_batch_i = y_t[idx].to(device)
                a_batch_i = a_t[idx].to(device)


                # Forward pass, compute loss, backpropagation
                z_pred = model(z_batch_i, a_batch_i)
                true_delta_z = (y_batch_i-z_batch_i[:, -1, :]).detach()
                scale = (0.5+torch.linalg.norm(true_delta_z, dim =-1))

                loss = torch.mean(scale * mse_loss(z_pred, y_batch_i).mean(dim=-1))

                loss.backward()

                #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # This is to avoid sudden changes in the parameters

                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            losses.append(epoch_loss / max(n_batches, 1))
            
            print(f"Epoch {epoch + 1}/{self.epochs} | Loss: {losses[-1]:.6f}")


        if save_model_as:
            torch.save(model.state_dict(), save_model_as)
            print(f"Model saved as {save_model_as}")

        self.losses = np.array(losses)
        
        return model, losses


    def plot_training_loss_vs_epoch(self):

        # Aggregate across ensemble models (2D array: num_models x epochs)
        if self.losses is not None and self.losses.ndim > 1:
            plot_data = np.mean(self.losses, axis=0)
        else:
            plot_data = self.losses

        plt.figure(figsize=(10, 5))
        plt.plot(plot_data, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Transition Model Training Loss')
        plt.show()

        return None


    def train_ensemble_transition_model(self, data_loader: 'TransitionDataLoader', ensemble_model: 'EnsembleTransitionModel', save_prefix = ""):

        """
        Trains each model in the ensemble on the same data.

        Parameters:
            ensemble_model: the ensemble model to train
            data_loader: To parse the data
            epochs times to repeat the training loop
            lr learning rate
            batch_size batches in which the training data is divided
        Returns:
            trained ensemble model
        """

        losses = []

        for i, model in enumerate(ensemble_model.models):
            print(f"Training model {i+1}/{ensemble_model.num_models}")

            model_name = f"{save_prefix}_transition_model_{i}.pth" if save_prefix else f"transition_model_{i}.pth"

            z_data, a_data, y_data = data_loader.load_full_dataset()

            model, loss = self.train_transition_model(z_data, a_data, y_data, model=model, save_model_as = model_name)
            
            losses.append(loss)

        self.losses = np.array(losses)

        return ensemble_model



    def train_ensmble_with_bagging(self, ensemble_model: 'EnsembleTransitionModel', data_loader: 'TransitionDataLoader', save_prefix = ""):
        """
        Trains an ensemble model using Bootstrap Aggregating (Bagging).

        For each model in the ensemble, a new dataset is constructed by randomly 
        sampling scenes with replacement from the full dataset. This increases 
        variance among the models, improving the ensemble's overall robustness.

        Parameters
        ----------
        ensemble_model : EnsembleTransitionModel
            The ensemble model containing the individual transition networks.
        data_loader : TransitionDataLoader
            The object that pases the training data.
        save_prefix : str
            Prefix for saving the individual model weights (e.g., hyperparameter ID).
        step_size : int
            The strided gap between consecutive measurements.
            
        Returns
        -------
        ensemble_model : EnsembleTransitionModel
            The fully trained ensemble.
        losses : np.ndarray
            Array containing the loss history for each individual model.
        """
        losses = []
        # Divide sequences into scenes (get scenes indices).
        scenes_indices = data_loader.generate_scene_indices()

        # Training for each model
        for i, model in enumerate(ensemble_model.models):

            print(f"Training model {i+1}/{ensemble_model.num_models}")
            model_name = f"{save_prefix}_transition_model_{i}.pth" if save_prefix else f"transition_model_{i}.pth"
            
            z_data, a_data, y_data = data_loader.load_from_sample_scenes_with_replacement(scenes_indices)

            model, loss = self.train_transition_model(z_data, a_data, y_data, model=model, save_model_as = model_name)
            
            losses.append(loss)

            del z_data, a_data, y_data
            gc.collect()

        self.losses = np.array(losses)

        return ensemble_model