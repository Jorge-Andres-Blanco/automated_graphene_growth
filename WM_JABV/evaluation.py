from pathlib import Path
from xml.parsers.expat import model
import os
from WM_JABV.transition_models import TransitionModel, EnsembleTransitionModel
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def evaluate_transition_model(model: TransitionModel, z_eval:np.ndarray, a_eval:np.ndarray, y_eval:np.ndarray):
    """
    Parameters:
        model: trained TransitionModel
        z_eval: np.ndarray (T, 384) — latent states
        a_eval: np.ndarray (T, 1) — action sequence
        y_eval: np.ndarray (T, 384) — observed states
    """

    #   # Compare y_pred vs y
    #   # Plot with PCA or UMAP
    #   # Compute per-step cosine similarity
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    z_eval = torch.from_numpy(z_eval).float().to(device)
    a_eval = torch.from_numpy(a_eval).float().to(device)
    y_eval = torch.from_numpy(y_eval).float().to(device)

    with torch.no_grad():

        y_pred = model(z_eval, a_eval)

        l2_distances = torch.linalg.norm(y_pred - y_eval, dim = -1)
        cos_similarities = nn.functional.cosine_similarity(y_pred, y_eval, dim=-1)
        mse_loss = nn.MSELoss()(y_pred, y_eval)

        # PCA

        pca = PCA(n_components=2)
        y_pca = pca.fit_transform(y_eval)
        y_pred_pca = pca.transform(y_pred)

    # Plot

    # PCA: one trajectory as a line
    plt.figure(figsize=(12, 5)) 
    plt.subplot(1, 3, 1)

    plt.scatter(y_pca[:, 0], y_pca[:, 1], color="blue", alpha=0.65, label="Real")
    plt.scatter(y_pred_pca[:, 0], y_pred_pca[:, 1], color="red", alpha=0.5, label="Predicted")
        
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(l2_distances, label='L2 Distance', color = 'red')
    plt.xlabel('Measurement')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(cos_similarities, label='Cosine Similarity')
    plt.xlabel('Measurement')
    plt.legend()

    plt.suptitle('Transition Model Evaluation')
    plt.show()

    return l2_distances, cos_similarities, mse_loss.item()


def evaluate_on_trajectory(model, z_traj, a_traj, y_traj):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    z_traj = torch.from_numpy(z_traj).float().to(device)
    a_traj = torch.from_numpy(a_traj).float().to(device)
    y_traj = torch.from_numpy(y_traj).float().to(device) # shape (time,384)

    time = z_traj.shape[0]
    l2_distances = np.zeros(time)
    cos_similarities = np.zeros(time)
    z_predicted_array = np.zeros((time, 384))
    
    h = model.history
    
    z = z_traj[0].unsqueeze(0) # Shape (1,hist,384)

    with torch.no_grad():
    
        for t in range(time):

            z_pred = model(z, a_traj[t].unsqueeze(0)) #shape (1,384)

            z = torch.cat((z[:,1:], z_pred.unsqueeze(0)), dim=1)

            z_predicted_array[t] = z_pred.squeeze(0).cpu().numpy()

            l2_distances[t] = torch.linalg.norm(z_pred-y_traj[t], dim = -1)

            cos_similarities[t] = nn.functional.cosine_similarity(z_pred, y_traj[t], dim=-1)

    pca = PCA(n_components=2)
    y_pca = pca.fit_transform(y_traj.cpu().numpy())
    y_pred_pca = pca.transform(z_predicted_array)

    return (y_pca, y_pred_pca), l2_distances, cos_similarities



def plot_trajectory_evaluation(y_pca, y_pred_pca, l2_distances, cos_similarities):

    # PCA: one trajectory as a line
    plt.figure(figsize=(12, 5)) 
    plt.subplot(1, 3, 1)

    plt.plot(y_pca[:, 0], y_pca[:, 1], color="blue", label="Real")
    plt.scatter(y_pca[:, 0], y_pca[:, 1], color="blue", alpha = 0.4, s = 20)
    
    plt.plot(y_pred_pca[:, 0], y_pred_pca[:, 1], color="red", label="Predicted")
    plt.scatter(y_pred_pca[:, 0], y_pred_pca[:, 1], color="red", alpha = 0.4, s=20)

    plt.scatter(y_pca[0, 0], y_pca[0, 1], color='black', s=20, label = "Start")
    plt.scatter(y_pred_pca[0, 0], y_pred_pca[0, 1], color='black', s=20)
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(l2_distances, label='L2 Distance', color = 'red')
    plt.xlabel('Steps')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(cos_similarities, label='Cosine Similarity')
    plt.xlabel('Steps')
    plt.legend()

    plt.suptitle('Transition Model Evaluation in a trajectory')
    plt.show()


def evaluate_ensemble_transition_model(ensemble_model: EnsembleTransitionModel, z_data: np.ndarray, a_data: np.ndarray, y_data: np.ndarray):

    """
    Parameters:
        ensemble_model: trained EnsembleTransitionModel
        z_data: np.ndarray (T, 384) — latent states
        a_data: np.ndarray (T, 2) — action sequence
        y_data: np.ndarray (T, 384) — observed states
    """

    #   # Compare y_pred vs y
    #   # Plot with PCA or UMAP
    #   # Compute per-step cosine similarity
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ensemble_model.to(device)
    ensemble_model.eval()

    z_eval = torch.from_numpy(z_data).float().to(device)
    a_eval = torch.from_numpy(a_data).float().to(device)
    y_eval = torch.from_numpy(y_data).float().to(device)

    with torch.no_grad():

        mean_pred, std_pred = ensemble_model.get_stats(z_eval, a_eval)

        l2_distances = torch.linalg.norm(mean_pred - y_eval, dim = -1)
        cos_similarities = nn.functional.cosine_similarity(mean_pred, y_eval, dim=-1)
        mse_loss = nn.MSELoss()(mean_pred, y_eval)

        # PCA

        pca = PCA(n_components=2)
        y_pca = pca.fit_transform(y_eval)
        y_pred_pca = pca.transform(mean_pred)

    # Plot

    # PCA: one trajectory as a line
    plt.figure(figsize=(12, 5)) 
    plt.subplot(1, 3, 1)

    plt.scatter(y_pca[:, 0], y_pca[:, 1], color="blue", alpha=0.65, label="Real")
    plt.scatter(y_pred_pca[:, 0], y_pred_pca[:, 1], color="red", alpha=0.5, label="Predicted")
        
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(l2_distances, label='L2 Distance', color = 'red')
    plt.xlabel('Measurement')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(cos_similarities, label='Cosine Similarity')
    plt.xlabel('Measurement')
    plt.legend()

    plt.suptitle('Transition Model Evaluation')
    plt.show()

    return l2_distances, cos_similarities, mse_loss.item()

def evaluate_on_trajectory(model, z_traj, a_traj, y_traj):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    z_traj = torch.from_numpy(z_traj).float().to(device)
    a_traj = torch.from_numpy(a_traj).float().to(device)
    y_traj = torch.from_numpy(y_traj).float().to(device) # shape (time,384)

    time = z_traj.shape[0]
    l2_distances = np.zeros(time)
    cos_similarities = np.zeros(time)
    z_predicted_array = np.zeros((time, 384))
    
    h = model.history
    
    z = z_traj[0].unsqueeze(0) # Shape (1,hist,384)

    with torch.no_grad():
    
        for t in range(time):

            z_pred = model(z, a_traj[t].unsqueeze(0)) #shape (1,384)

            z = torch.cat((z[:,1:], z_pred.unsqueeze(0)), dim=1)

            z_predicted_array[t] = z_pred.squeeze(0).cpu().numpy()

            l2_distances[t] = torch.linalg.norm(z_pred-y_traj[t], dim = -1)

            cos_similarities[t] = nn.functional.cosine_similarity(z_pred, y_traj[t], dim=-1)

    pca = PCA(n_components=2)
    y_pca = pca.fit_transform(y_traj.cpu().numpy())
    y_pred_pca = pca.transform(z_predicted_array)

    return (y_pca, y_pred_pca), l2_distances, cos_similarities


def evaluate_ensemble_on_trajectory(ensemble_model: EnsembleTransitionModel, z_traj: np.ndarray, a_traj: np.ndarray, y_traj: np.ndarray):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ensemble_model.to(device)
    ensemble_model.eval()

    z_traj = torch.from_numpy(z_traj).float().to(device)
    a_traj = torch.from_numpy(a_traj).float().to(device)
    y_traj = torch.from_numpy(y_traj).float().to(device) # shape (time,384)

    time = z_traj.shape[0]
    l2_distances = np.zeros(time)
    cos_similarities = np.zeros(time)
    z_predicted_array = np.zeros((time, 384))
    
    z = z_traj[0].unsqueeze(0) # Shape (1,hist,384)

    with torch.no_grad():
    
        for t in range(time):

            z_pred, std_pred = ensemble_model.get_stats(z, a_traj[t].unsqueeze(0)) #shape (num_models, 1,384)

            z_mean = z_pred.mean(dim=0, keepdim=True) #shape (1,384)

            z = torch.cat((z[:,1:], z_mean.unsqueeze(0)), dim=1)

            z_predicted_array[t] = z_mean.squeeze(0).cpu().numpy()

            l2_distances[t] = torch.linalg.norm(z_mean-y_traj[t], dim = -1)

            cos_similarities[t] = nn.functional.cosine_similarity(z_mean, y_traj[t], dim=-1)

    pca = PCA(n_components=2)
    y_pca = pca.fit_transform(y_traj.cpu().numpy())
    y_pred_pca = pca.transform(z_predicted_array)

    return (y_pca, y_pred_pca), l2_distances, cos_similarities


def predict_next_steps(steps: int, model: TransitionModel, z_init: np.ndarray, a_init: np.ndarray, a_future: np.ndarray):
    """
    Generates open-loop autoregressive predictions for state transitions over a given number of future steps.

    Parameters
    ----------
    steps : int
        The number of future steps to predict.
    model : nn.Module or EnsembleTransitionModel
        The transition model used for prediction. Can be a single model or an ensemble.
    z_init : np.ndarray
        The initial latent state history. Expected shape: (history, latent_dim) or (latent_dim,).
    a_init : np.ndarray
        The initial action history that led to `z_init`. Expected shape: (history,) or (history, action_dim).
    a_future : np.ndarray
        The intended future actions for the next `steps`. Expected shape: (steps,) or (steps, action_dim).

    Returns
    -------
    predictions : torch.Tensor
        The predicted mean latent states for the next `steps`. Shape: (steps, latent_dim).
    std_predictions : torch.Tensor or None
        The standard deviation of the predictions if the model is an ensemble, shape (steps, latent_dim). None otherwise.

    Hardcoded Elements
    ------------------
    - Latent dimension: Hardcoded to 384 during the pre-allocation of the `predictions` and `std_predictions` tensors (e.g., `torch.zeros((steps, 384))`).
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    predictions = torch.zeros((steps, 384), device=device)
    is_ensemble = hasattr(model, 'num_models') or isinstance(model, EnsembleTransitionModel)
    
    if is_ensemble:
        std_predictions = torch.zeros((steps, 384), device=device)
    else:
        std_predictions = None

    # Ensure z_init has the correct 3D shape (1, history, 384)
    z_init_t = torch.from_numpy(z_init).float().to(device)
    if z_init_t.dim() == 1:
        z_init_t = z_init_t.unsqueeze(-1)
    z_hist = z_init_t.unsqueeze(0) 

    # Ensure a_init has the trailing action_dim (history, action_dim) -> (1, history, action_dim)
    a_init_t = torch.from_numpy(a_init).float().to(device)
    if a_init_t.dim() == 1:
        a_init_t = a_init_t.unsqueeze(-1) 
    a_hist = a_init_t.unsqueeze(0) 


    a_fut = torch.from_numpy(a_future).float().to(device)
    if a_fut.dim() == 1:
        a_fut = a_fut.unsqueeze(-1)

    with torch.no_grad():

        for t in range(steps):

            if is_ensemble:
                # Based on your logic, extract mean and std 
                z_mean, z_std = model.get_stats(z_hist, a_hist)
                
                predictions[t] = z_mean.squeeze(0)
                std_predictions[t] = z_std.squeeze(0)
                z_next = z_mean
            else:
                z_next = model(z_hist, a_hist)
                predictions[t] = z_next.squeeze(0)

            # Update sliding windows by dropping oldest and appending newest
            z_hist = torch.cat((z_hist[:, 1:, :], z_next.unsqueeze(0)), dim=1)
            a_hist = torch.cat((a_hist[:, 1:, :], a_fut[t].unsqueeze(0).unsqueeze(0)), dim=1)

    return predictions, std_predictions



def extract_delta_z(z_init: np.ndarray, predictions: torch.Tensor) -> torch.Tensor:
    """
    Extracts the step-by-step state changes (delta z) from a generated sequence of predictions.

    Parameters
    ----------
    z_init : np.ndarray
        The initial observed latent state or history of states. Expected shape: (history_length, latent_dim) or (latent_dim,).
    predictions : torch.Tensor
        The predicted future states over `steps`. Expected shape: (steps, latent_dim).

    Returns
    -------
    delta_z : torch.Tensor
        The step-by-step differences representing (z_t - z_{t-1}). Shape strictly matches the `predictions` tensor: (steps, latent_dim).
    """

    device = predictions.device

    z_last = z_init[-1] if z_init.ndim > 1 else z_init
    z_last = torch.from_numpy(z_last).float().to(device) # shape (384,)

    # Trajectory shape: (1 + steps, 384)
    trajectory = torch.cat([z_last.unsqueeze(0), predictions], dim=0)
    delta_z = trajectory[1:] - trajectory[:-1]

    return delta_z




def evaluate_rollouts(steps: int, model, z_traj: np.ndarray, a_traj: np.ndarray, y_traj: np.ndarray):
    """
    Evaluates an N-step open-loop rollout across an entire trajectory by comparing model predictions against ground truth observations.

    Parameters
    ----------
    steps : int
        The number of forward prediction steps for each evaluation window.
    model : nn.Module or EnsembleTransitionModel
        The transition model being evaluated.
    z_traj : np.ndarray
        The true pre-windowed latent states from the dataloader. Expected shape: (N, history, latent_dim).
    a_traj : np.ndarray
        The true pre-windowed actions from the dataloader. Expected shape: (N, history, action_dim).
    y_traj : np.ndarray
        The true target latent states for evaluation. Expected shape: (N, latent_dim).

    Returns
    -------
    dz_mean : np.ndarray
        The mean step-by-step displacement magnitude across all evaluation windows. Shape: (steps, latent_dim).
    std_z_mean : np.ndarray
        The mean step-by-step standard deviation (uncertainty) across all evaluation windows. Shape: (steps, latent_dim).
    l2_distances_mean : np.ndarray
        The mean L2 distance (error) between predictions and targets per rollout step. Shape: (steps,).
    cos_similarities_mean : np.ndarray
        The mean cosine similarity between predictions and targets per rollout step. Shape: (steps,).

    Hardcoded Elements
    ------------------
    - Latent dimension: Hardcoded to 384 during the pre-allocation of the `dz` and `std_z` NumPy arrays (e.g., `np.zeros((num_evals, steps, 384))`).
    """
        
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()
    history = model.models[0].history if hasattr(model, 'num_models') else model.history
    time_len = z_traj.shape[0]

    # Calculate how many full evaluation windows fit in the trajectory
    num_evals = time_len - history - steps + 1

    dz = np.zeros((num_evals, steps, 384))
    std_z = np.zeros((num_evals, steps, 384))
    l2_distances = np.zeros((num_evals, steps))
    cos_similarities = np.zeros((num_evals, steps))
    mse_losses = np.zeros((num_evals, steps))

    y_traj_t = torch.from_numpy(y_traj).float().to(device) # shape (time,384)

    with torch.no_grad():
    
        for i, t in enumerate(range(num_evals)):
            
            # z_traj and a_traj are ALREADY windowed from the dataloader
            z_init = z_traj[t] # Shape: (history, 384)
            a_init = a_traj[t] # Shape: (history, action_dim)
            
            # Extract future actions by taking the newest action from the subsequent overlapping windows
            a_future = np.array([a_traj[t + k][-1] for k in range(1, steps + 1)]) # Shape: (steps, action_dim)
            
            y_target = y_traj_t[t:t+steps] # Shape: (steps, 384)
            
            # prediction
            predictions, std_predictions = predict_next_steps(steps, model, z_init, a_init, a_future)
            
            dz[i] = extract_delta_z(z_init, predictions).cpu().numpy()
            
            if std_predictions is not None:
                std_z[i] = std_predictions.cpu().numpy()
                
            l2_distances[i] = torch.linalg.norm(predictions - y_target, dim=-1).cpu().numpy()
            cos_similarities[i] = torch.nn.functional.cosine_similarity(predictions, y_target, dim=-1).cpu().numpy()
            mse_losses[i] = torch.nn.functional.mse_loss(predictions, y_target, reduction='none').mean(dim=-1).cpu().numpy()

    return dz.mean(axis = 0), std_z.mean(axis = 0), l2_distances.mean(axis = 0), cos_similarities.mean(axis = 0)



def plot_uncertainty_ratio(data_dir):

    """
    Reads rollout evaluation metrics from a specified directory and generates a 2x2 grid plot 
    of dz/std(z) of different models over prediction steps.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        The absolute or relative path to the directory containing the `.npz` metric files.

    Returns
    -------
    None
        Generates and saves a `.png` plot (`uncertainty_ratio_plot.png`) to the specified `data_dir`.

    Hardcoded Elements
    ------------------
    - `hist_range`: Hardcoded to [1,2,5,10,15, 20, 25].
    - `step_size_range`: Hardcoded to [2, 4, 5, 7, 10].
    - `rollout_steps`: Hardcoded to 5.
    - Plot styling
    """
    


    hist_range = [1, 2, 5, 10, 15, 20, 25]
    step_size_range = [2,4,5,7]
    rollout_steps = 5 
    
    # Create a 2x2 grid of plots for the 4 different step sizes
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, step_size in enumerate(step_size_range):
        ax = axes[idx]
        
        for hist in hist_range:
            run_id = f"hist_{hist}_step_{step_size}"
            filename = f"{data_dir}rollout_metrics_{run_id}.npz"
            
            if not os.path.exists(filename):
                print(f"Warning: File {filename} not found. Skipping.")
                continue
                
            
            data = np.load(filename)
            dz = data['dz']       # Shape: (steps, 384)
            std_z = data['std_z'] # Shape: (steps, 384)
            
            
            # Calculate the magnitude of the displacement
            dz_norm = np.linalg.norm(dz, axis=1)  # Shape: (steps,)
            
            # Calculate the magnitude of the uncertainty
            std_norm = np.linalg.norm(std_z, axis=1)  # Shape: (steps,)
            
            ratio_scalar = dz_norm / (std_norm + 1e-8)
            
            x_axis = np.arange(1, rollout_steps + 1) * step_size*2
            ax.plot(x_axis, ratio_scalar, marker='o', linewidth=2, label=f'History: {hist}')

        
        ax.set_title(f"Model Step Size: {step_size}")
        ax.set_xlabel("Time in advance of prediction (s)")
        ax.set_ylabel(r"Norm Ratio $(||\Delta z||_2 / ||\sigma_z||_2)$")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

    plt.suptitle("Transition Model Uncertainty Calibration: Signal-to-Noise Ratio", fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save and display
    plt.savefig("uncertainty_ratio_plot.png", dpi=300, bbox_inches='tight')
    plt.show()



def plot_time_comparisons(data_dir: str | Path):

    """
    Reads rollout evaluation metrics from a specified directory and generates a grouped bar chart 
    comparing dz/std(z).

    Parameters
    ----------
    data_dir : str or pathlib.Path
        The absolute or relative path to the directory containing the `.npz` metric files.

    Returns
    -------
    None
        Generates and saves a `.png` plot to the specified `data_dir`.

    Hardcoded Elements
    ------------------
    - `hist_range`: Hardcoded to [1,2,5,10,15, 20, 25].
    - `step_size_range`: Hardcoded to [2,4,5,7].
    - `rollout_steps`: Hardcoded to 5.
    - Target time horizons: Hardcoded to 20s and 40s for the interpolation and bar chart groupings.
    - Plot styling
    """
    data_dir = Path(data_dir)
    hist_range = [1, 2, 5, 10, 15, 20, 25]
    step_size_range = [2,4,5,7]
    rollout_steps = 5 
    
    time_multiplier = 2 
    
    # Structure: {time: {history: [val_step4, val_step5, val_step7, val_step10]}}
    results = {
        20: {h: [] for h in hist_range},
        40: {h: [] for h in hist_range}
    }

    for hist in hist_range:
        for step_size in step_size_range:
            run_id = f"hist_{hist}_step_{step_size}"
            filepath = data_dir / f"rollout_metrics_{run_id}.npz"
            
            if not filepath.exists():
                print(f"File {filepath} not found. Inserting NaN.")
                # Insert NaN so the bar chart leaves an empty space instead of crashing
                results[20][hist].append(np.nan)
                results[40][hist].append(np.nan)
                continue
                
            data = np.load(filepath)
            dz = data['dz']       
            std_z = data['std_z'] 
            
            dz_norm = np.linalg.norm(dz, axis=1)  
            std_norm = np.linalg.norm(std_z, axis=1)  
            ratio_scalar = dz_norm / (std_norm + 1e-8)
            
            time_axis = np.arange(1, rollout_steps + 1) * step_size * time_multiplier
            
            val_20 = np.interp(20, time_axis, ratio_scalar)
            val_40 = np.interp(40, time_axis, ratio_scalar)
            
            results[20][hist].append(val_20)
            results[40][hist].append(val_40)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    x = np.arange(len(step_size_range))  # the label locations
    width = 0.25  # the width of the bars

    for idx, target_time in enumerate([20, 40]):
        ax = axes[idx]
        
        ax.bar(x - width, results[target_time][15], width, label='History: 15', color='#1f77b4') # Blue
        ax.bar(x,         results[target_time][20], width, label='History: 20', color='#ff7f0e') # Orange
        ax.bar(x + width, results[target_time][25], width, label='History: 25', color='#2ca02c') # Green

        ax.set_title(f"Model Performance at {target_time}s Prediction Horizon")
        ax.set_xlabel("Model Step Size")
        ax.set_ylabel(r"Norm Ratio $(||\Delta z||_2 / ||\sigma_z||_2)$")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Step: {s}" for s in step_size_range])
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.legend()

    plt.suptitle("Direct Model Comparison at Specific Time Horizons", fontsize=16)
    plt.tight_layout()    
    plt.show()


def plot_evaluation_metrics(data_dir: str | Path):
    """
    Reads rollout evaluation metrics from a specified directory and generates two separate 2x2 grid plots 
    visualizing the L2 Distances (Error) and Cosine Similarities (Alignment) over prediction steps.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        The absolute or relative path to the directory containing the `.npz` metric files.

    Returns
    -------
    None
        Generates and saves two `.png` plots (`l2_distance_plot.png`, `cosine_similarity_plot.png`) to the specified `data_dir`.

    Hardcoded Elements
    ------------------
    - `hist_range`: Hardcoded to [1,2,5,10,15, 20, 25].
    - `step_size_range`: Hardcoded to [2,4,5,7].
    - `rollout_steps`: Hardcoded to 5.
    - Plot styling:
    """
    data_dir = Path(data_dir)
    
    hist_range = [1, 2, 5, 10, 15, 20, 25]
    step_size_range = [2,4,5,7]
    rollout_steps = 5 
    
    fig_l2, axes_l2 = plt.subplots(2, 2, figsize=(14, 10))
    axes_l2 = axes_l2.flatten()
    
    fig_cos, axes_cos = plt.subplots(2, 2, figsize=(14, 10))
    axes_cos = axes_cos.flatten()

    for idx, step_size in enumerate(step_size_range):
        ax_l2 = axes_l2[idx]
        ax_cos = axes_cos[idx]
        
        for hist in hist_range:
            run_id = f"hist_{hist}_step_{step_size}"
            filepath = data_dir / f"rollout_metrics_{run_id}.npz"
            
            if not filepath.exists():
                print(f"Warning: File {filepath} not found. Skipping.")
                continue
                
            data = np.load(filepath)
            
            l2_dist = data['l2_distances'] 
            cos_sim = data['cos_similarities']
            
            x_axis = np.arange(1, rollout_steps + 1) * step_size*2
            
            ax_l2.plot(x_axis, l2_dist, marker='o', linewidth=2, label=f'History: {hist}')
            ax_cos.plot(x_axis, cos_sim, marker='s', linewidth=2, label=f'History: {hist}')

        ax_l2.set_title(f"Model Step Size: {step_size}")
        ax_l2.set_xlabel("Time in advance of prediction (s)")
        ax_l2.set_ylabel("Mean L2 Distance (Error)")
        ax_l2.grid(True, linestyle='--', alpha=0.7)
        ax_l2.legend()

        ax_cos.set_title(f"Model Step Size: {step_size}")
        ax_cos.set_xlabel("Time in advance of prediction (s)")
        ax_cos.set_ylabel("Mean Cosine Similarity")
        ax_cos.grid(True, linestyle='--', alpha=0.7)
        ax_cos.legend()

    fig_l2.suptitle("Transition Model Error: L2 Distance vs Time", fontsize=16, y=1.02)
    fig_l2.tight_layout()
    
    fig_cos.suptitle("Transition Model Alignment: Cosine Similarity vs Time", fontsize=16, y=1.02)
    fig_cos.tight_layout()
    
    plt.show()




def plot_possible_actions_losses(losses:torch.Tensor, actions: torch.Tensor, aggregate='mean', save_path=None):
    """
    Visualizes the predictive losses for different constant-action sequences.

    Parameters
    ----------
    losses : torch.Tensor or np.ndarray
        The distance metric for each action sequence. Shape: (possibilities, num_models, steps).
    actions : torch.Tensor, np.ndarray, or list
        The corresponding action values that were evaluated. Length: (possibilities,).
    aggregate : str or None
        If 'mean', plots a single bar per action representing the loss averaged 
        across all future steps. If None, plots a grouped bar chart showing the 
        loss at each specific future step.
    save_path : str or pathlib.Path, optional
        If provided, saves the figure to this location.
    """    

    # Force inputs into standard NumPy arrays for matplotlib compatibility
    losses = losses.cpu().numpy()
    actions = actions.cpu().numpy()
        
    num_actions, num_models, steps = losses.shape
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    if aggregate == 'mean':
        # Average across the temporal rollout (steps) first
        # Shape becomes (possibilities, num_models)
        time_averaged_losses = losses.mean(axis=2)
        
        # 2. Calculate the mean and std across the ensemble models (num_models)
        # Shapes become (possibilities,)
        mean_losses = time_averaged_losses.mean(axis=1)
        std_losses = time_averaged_losses.std(axis=1)

        # Color the best (lowest loss) action distinctly to make the chart actionable
        best_idx = np.argmin(mean_losses)
        colors = ['#1f77b4' if i != best_idx else '#2ca02c' for i in range(num_actions)]
        
        ax.bar(
            actions.astype(str), 
            mean_losses, 
            yerr=std_losses,
            ecolor='red', 
            capsize=5,             # Adds horizontal caps to the error bars
            color=colors, 
            edgecolor='black',
            alpha=0.85
        )
        
        ax.set_ylabel("Mean Loss (MSE + Alignment)", fontsize = 14)
        ax.set_title("Average Predictive Loss per Action Sequence", fontsize = 16)
        
    elif aggregate is None:
        # Shapes become (possibilities, steps)
        mean_losses = losses.mean(axis=1)
        std_losses = losses.std(axis=1)
        
        x_indices = np.arange(num_actions)
        bar_width = 0.8 / steps
        
        # Calculate offsets to perfectly center the group of bars over the tick mark
        offsets = np.linspace(-0.4 + bar_width/2, 0.4 - bar_width/2, steps)
        
        for s in range(steps):
            ax.bar(
                x_indices + offsets[s], 
                mean_losses[:, s], 
                yerr=std_losses[:, s], # Apply the step-specific standard deviation
                width=bar_width, 
                capsize=3,             # Smaller caps for crowded grouped bars
                label=f'Step {s+1}',
                alpha=0.9,
                edgecolor='black'
            )
        ax.set_ylabel("Step-wise Loss", fontsize = 14)
        ax.set_title("Temporal Predictive Loss per Action (Error bars = Ensemble Std)", fontsize = 16)
        ax.legend(title="Rollout Step", fontsize = 12)
        
    else:
        raise ValueError("Invalid aggregate parameter. Use 'mean' or None.")
    
    tick_positions = [i for i, action in enumerate(actions) if action % 1 == 0]
    
    # 2. Extract those exact integer values to use as the visual labels
    tick_labels = [int(actions[i]) for i in tick_positions]
    
    # 3. Apply them to the axis
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(-0.5,num_actions-0.5)
    ax.tick_params(axis='both', labelsize = 12)
    ax.set_xlabel("Constant CH4 Action Value", fontsize = 14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        
    plt.show()



@torch.no_grad()
def plot_actions_vs_time_for_sequence(ensemble_model, z_sequence, a_sequence, step_size, history, a_pos="all", future_steps=5, save_path=None):
    """
    Simulates a sequence of frames, predicting the optimal action to reach a future 
    target dynamically extracted from the trajectory.
    """
    #device = next(model.parameters()).device

    N = z_sequence.shape[0]

    # Calculate the offsets at the begining and end of the plot
    horizon_offset = future_steps*step_size
    context_offset = step_size * history


    # Only evaluate frames where a future target exists
    valid_frames = N - horizon_offset
    
    if valid_frames <= 0:
        raise ValueError(f"sequence length ({N}) is too short to evaluate a horizon of {future_steps} steps.")

    # Pre-allocate arrays for plotting
    pred_actions = np.zeros(valid_frames)
    std_actions = np.zeros(valid_frames)
    real_actions = np.zeros(valid_frames)

    for i in range(valid_frames):
        
        # Current sliding window
        z_init = z_sequence[i]#.clone().to(device)
        a_init = a_sequence[i]#.clone().to(device)

        # z_sequence[i + steps] is the future sliding window. 
        # [-1] to get that specific future frame (not the whole window).
        target = z_sequence[i + horizon_offset][-1]#.clone().to(device)

        # Real action that was actually executed at time t
        real_actions[i] = a_init[-1].item()

        # Predict the best action to reach the future target (t+step)
        pred_a, std_a = ensemble_model.predict_next_step(
            steps=future_steps,
            z_init=z_init,
            a_init=a_init,
            a_pos=a_pos,
            target=target
        )

        pred_actions[i] = pred_a
        std_actions[i] = std_a

    # --- PLOTTING ---
    frames = np.arange(valid_frames) + context_offset
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(frames, real_actions, label='Real', color='black', linewidth=2, linestyle='--')
    ax.plot(frames, pred_actions, label=f'Predicted Optimal (to reach target)', color='#1f77b4', linewidth=2)

    ax.fill_between(
        frames,
        pred_actions - std_actions,
        pred_actions + std_actions,
        color="#b41f1f",
        alpha=0.1,
        label=r'Ensemble Uncertainty (±1 $\sigma$)'
    )
    ax.set_ylim(0,20)
    ax.set_xlim(frames[0], frames[-1])
    ax.set_xlabel("Frame", fontsize = 14)
    ax.set_ylabel("CH4 Flow (sccm)", fontsize = 14)
    ax.set_title(f"Real vs Predicted Actions (Target = {future_steps} steps ahead), Possible Actions: {a_pos}", fontsize = 16)
    ax.tick_params(axis='both', labelsize = 12)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=':', alpha=0.7)
    
    ax.xaxis.get_major_locator().set_params(integer=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()