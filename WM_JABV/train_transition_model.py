import torch
import torch.nn as nn
from WM_JABV.transition_models import TransitionModel, EnsembleTransitionModel
from data_processing.data_loader import *
import numpy as np
import matplotlib.pyplot as plt
import gc


def train_transition_model(z_data:np.ndarray, a_data:np.ndarray, y_data: np.ndarray,
                           model: TransitionModel,  epochs: int=50, lr: float=1e-3,
                           batch_size: int=64, save_model_as: str= "transition_model.pth") -> TransitionModel:
    
    """
    Trains the transition model on collected data.

    Parameters:
        z_data shape (batch_size, history, latent_dim)
        a_data shape (batch_size, history, action_dim)
        model to train
        y_data shape (batch_size, latent_dim)
        epochs times to repeat the training loop
        lr learning rate
        batch_size batches in which the training data is divided
        save_model_as model name (.pth), empty to not save

    Returns:
        trained model
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")


    model.to(device)
    model.train()

    z_h = torch.from_numpy(z_data).float()
    a_h = torch.from_numpy(a_data).float()
    y = torch.from_numpy(y_data).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_samples = z_h.shape[0]
    
    losses = []

    mse_loss = nn.MSELoss()


    for epoch in range(epochs):


        # Shuffle
        perm = torch.randperm(n_samples)

        epoch_loss = 0.0
        n_batches = 0       

        for i in range(0, n_samples, batch_size):

            optimizer.zero_grad()

            idx = perm[i:i + batch_size]            

            # Current batch to device
            z_n_i = z_h[idx].to(device)
            y_n_i = y[idx].to(device)
            a_n_i = a_h[idx].to(device)


            # Forward pass, compute loss, backpropagation
            z_pred = model(z_n_i, a_n_i)

            loss = mse_loss(z_pred, y_n_i)

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # This is to avoid sudden changes in the parameters

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        losses.append(epoch_loss / max(n_batches, 1))
        
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {losses[-1]:.6f}")


    if save_model_as:
        torch.save(model.state_dict(), save_model_as)
        print(f"Model saved as {save_model_as}")

    return model, losses


def plot_training_loss(losses):

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Transition Model Training Loss')
    plt.show()

    return None

def train_ensemble_transition_model(z_data, a_data, y_data, ensemble_model: EnsembleTransitionModel, save_prefix = "", **kwargs):

    """
    Trains each model in the ensemble on the same data.

    Parameters:
        ensemble_model: the ensemble model to train
        z_data shape (batch_size, history, latent_dim)
        a_data shape (batch_size, history, action_dim)
        y_data shape (batch_size, latent_dim)
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

        model, loss = train_transition_model(z_data, a_data, y_data, model=model, save_model_as = model_name, **kwargs)
        
        losses.append(loss)

    return ensemble_model, np.array(losses)



def train_ensmble_with_bagging(ensemble_model: EnsembleTransitionModel, data_path = Path(r"/data/lmcat/Computer_vision/training_data"), save_prefix = "", step_size = 5, **kwargs):
    """
    Trains an ensemble model using Bootstrap Aggregating (Bagging).

    For each model in the ensemble, a new dataset is constructed by randomly 
    sampling scenes with replacement from the full dataset. This increases 
    variance among the models, improving the ensemble's overall robustness.

    Parameters
    ----------
    ensemble_model : EnsembleTransitionModel
        The ensemble model containing the individual transition networks.
    file_path : pathlib.Path
        The directory path containing the `*sequence*.npy` and `*CH4*.npy` files.
    save_prefix : str
        Prefix for saving the individual model weights (e.g., hyperparameter ID).
    step_size : int
        The strided gap between consecutive measurements.
    **kwargs 
        Additional arguments (epochs, lr, batch_size) passed to the core trainer.
        
    Returns
    -------
    ensemble_model : EnsembleTransitionModel
        The fully trained ensemble.
    losses : np.ndarray
        Array containing the loss history for each individual model.
    """
    losses = []
    history = ensemble_model.models[0].history

    # Divide sequences into scenes (get scenes indices).
    cls_files = sorted(data_path.glob("*sequence*.npy"), key=(lambda p: int(p.stem.split('_')[-1])))
    scenes_indices = get_scenes_indices_from_files(cls_files, history, step_size)

    # Training for each model
    for i, model in enumerate(ensemble_model.models):

        print(f"Training model {i+1}/{ensemble_model.num_models}")
        model_name = f"{save_prefix}_transition_model_{i}.pth" if save_prefix else f"transition_model_{i}.pth"
        
        z_data, a_data, y_data = load_transition_data_from_scene(data_path, scenes_indices, history, step_size)

        model, loss = train_transition_model(z_data, a_data, y_data, model=model, save_model_as = model_name, **kwargs)
        
        losses.append(loss)

        del z_data, a_data, y_data
        gc.collect()

    return ensemble_model, np.array(losses)