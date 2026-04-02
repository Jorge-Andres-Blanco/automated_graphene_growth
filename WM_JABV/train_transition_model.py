import torch
import torch.nn as nn
from WM_JABV.transition_model import TransitionModel
import numpy as np



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

    return model