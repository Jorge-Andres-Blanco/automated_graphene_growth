from pathlib import Path
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure


def plot_2_frames(frame_0, frame_1):
    
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(frame_0)
    ax[1].imshow(frame_1)

    plt.show()

    return None

def compare_images_in_latent_space(img1, img2, z1, z2, cmap='gray'):
    """
    Plots two images side-by-side and displays their L2 distance 
    and cosine similarity in the latent space.

    Args:
        img1 (np.ndarray): The first image array (e.g., from the Basler camera).
        img2 (np.ndarray): The second image array.
        z1 (np.ndarray): The DINOv2 latent embedding for the first image. Shape: (384,)
        z2 (np.ndarray): The DINOv2 latent embedding for the second image. Shape: (384,)
        cmap (str, optional): Colormap for matplotlib. Defaults to 'gray'.
    """
    
    # Ensure the embeddings are 1D arrays
    z1 = np.squeeze(z1)
    z2 = np.squeeze(z2)

    # Calculate L2 Distance
    l2_distance = np.linalg.norm(z1 - z2)

    # Calculate Cosine Similarity
    # Formula: (A dot B) / (||A|| * ||B||)
    dot_product = np.dot(z1, z2)
    norm_z1 = np.linalg.norm(z1)
    norm_z2 = np.linalg.norm(z2)
    
    # Avoid division by zero just in case
    if norm_z1 == 0 or norm_z2 == 0:
        cos_similarity = 0.0
    else:
        cos_similarity = dot_product / (norm_z1 * norm_z2)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Image 1
    axes[0].imshow(img1, cmap=cmap)
    axes[0].set_title("Image 1", fontsize=14)
    axes[0].axis('off') # Hide axes for cleaner image view

    # Plot Image 2
    axes[1].imshow(img2, cmap=cmap)
    axes[1].set_title("Image 2", fontsize=14)
    axes[1].axis('off')

    # Add the metrics as a main title above the images
    title_text = (
        f"Latent Space Comparison\n"
        f"L2 Distance: {l2_distance:.4f}  |  Cosine Similarity: {cos_similarity:.4f}"
    )
    plt.suptitle(title_text, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"comparison_img_and_tokens.png")


def plot_uncertainty_ratio(data_dir: str | Path):

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
    data_dir = Path(data_dir)


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
            
        Returns
        -------
        None
            Displays or saves the generated plot.
            
        Hardcoded Elements
        ------------------
        - Plot styling: Figure size (14, 6), custom hex colors for best action vs others.
        - Visualization limits: x-axis limits dynamically set using `-0.5` offsets.
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



def plot_actions_vs_time_for_sequence(ensemble_model, z_sequence, a_sequence, step_size, history, a_pos="all", future_steps=5, save_path=None):
    """
    Simulates a sequence of frames, predicting the optimal action to reach a future 
    target dynamically extracted from the trajectory.
    
    Parameters
    ----------
    ensemble_model : EnsembleTransitionModel
        The trained ensemble model used to predict action losses.
    z_sequence : np.ndarray or torch.Tensor
        The sliding-windowed sequence of latent states. Shape: (N, history, 384).
    a_sequence : np.ndarray or torch.Tensor
        The sliding-windowed sequence of actions. Shape: (N, history, action_dim).
    step_size : int
        The step size between sequence frames.
    history : int
        The history length used in each window.
    a_pos : str, optional
        The action space evaluation strategy. Defaults to "all".
    future_steps : int, optional
        The lookahead horizon for predictions. Defaults to 5.
    save_path : str or pathlib.Path, optional
        If provided, saves the generated plot to this path.
        
    Returns
    -------
    None
        Displays or saves the plotted actions over time.
        
    Hardcoded Elements
    ------------------
    - Plot styling: Figure size (14, 6), axis limits (y-axis limited to 0-20),
    fill opacity (alpha=0.1), colors, labels, and grid settings.
    - Offsets computation: Uses `future_steps * step_size` and `step_size * history`
    to calculate the frame alignment between current context and future targets.
    """
    #device = next(model.parameters()).device
    N = z_sequence.shape[0]

    # Calculate the offsets at the begining and end of the plot
    horizon_offset = future_steps*step_size
    context_offset = step_size * (history-1)


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


def plot_possible_actions_losses(losses:torch.Tensor, actions: torch.Tensor, aggregate='mean', ax=None, save_path=None, show=True):
    """
    Visualizes the predictive losses for different constant-action sequences.

    Parameters
    ----------
    losses : torch.Tensor or np.ndarray
        The distance metric for each action sequence. Shape: (possibilities, num_models, planning_horizon).
    actions : torch.Tensor, np.ndarray, or list
        The corresponding action values that were evaluated. Length: (possibilities,).
    aggregate : str or None
        If 'mean', plots a single bar per action representing the loss averaged 
        across all future steps (planning_horizon). If None, plots a grouped bar chart showing the 
        loss at each specific future step.
    save_path : str or pathlib.Path, optional
        If provided, saves the figure to this location.
    show : bool, optional
        If True, displays the figure. If False, only saves it.

        Returns
        -------
        None
            Displays or saves the generated plot.
            
        Hardcoded Elements
        ------------------
        - Plot styling: Figure size (14, 6), custom hex colors for best action vs others.
        - Visualization limits: x-axis limits dynamically set using `-0.5` offsets.
    """    

    # Force inputs into standard NumPy arrays for matplotlib compatibility
    losses = losses.cpu().numpy()
    actions = actions.cpu().numpy()
        
    num_actions, num_models, steps = losses.shape
    
    if ax is None:
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
        
        ax.set_ylabel("Planning Loss", fontsize = 18)
        ax.set_title(r"Planning Loss per CH$_4$ Flow Rate", fontsize = 20)
        
    elif aggregate is None:
        # THIS IS INCOMPLETE
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
    ax.tick_params(axis='both', labelsize = 16)
    ax.set_xlabel(r"Constant CH$_4$ Flow Rate (sccm)", fontsize = 18)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()


def plot_actions_vs_time_for_sequence(ensemble_model, z_sequence, a_sequence, step_size, history, a_pos="all", future_steps=5, save_path=None):
    """
    Simulates a sequence of frames, predicting the optimal action to reach a future 
    target dynamically extracted from the trajectory.
    
    Parameters
    ----------
    ensemble_model : EnsembleTransitionModel
        The trained ensemble model used to predict action losses.
    z_sequence : np.ndarray or torch.Tensor
        The sliding-windowed sequence of latent states. Shape: (N, history, 384).
    a_sequence : np.ndarray or torch.Tensor
        The sliding-windowed sequence of actions. Shape: (N, history, action_dim).
    step_size : int
        The step size between sequence frames.
    history : int
        The history length used in each window.
    a_pos : str, optional
        The action space evaluation strategy. Defaults to "all".
    future_steps : int, optional
        The lookahead horizon for predictions. Defaults to 5.
    save_path : str or pathlib.Path, optional
        If provided, saves the generated plot to this path.
        
    Returns
    -------
    None
        Displays or saves the plotted actions over time.
        
    Hardcoded Elements
    ------------------
    - Plot styling: Figure size (14, 6), axis limits (y-axis limited to 0-20),
    fill opacity (alpha=0.1), colors, labels, and grid settings.
    - Offsets computation: Uses `future_steps * step_size` and `step_size * history`
    to calculate the frame alignment between current context and future targets.
    """
    #device = next(model.parameters()).device
    N = z_sequence.shape[0]

    # Calculate the offsets at the begining and end of the plot
    horizon_offset = future_steps*step_size
    context_offset = step_size * (history-1)


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

def plot_uncertainty_ratio(data_dir: str | Path):

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
    data_dir = Path(data_dir)


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


def adjust_exposure_gray_image(img):
    """
    Adjusts the exposure of a grayscale image using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Parameters
    ----------
    img : np.ndarray
        A 2D array representing the grayscale image.

    Returns
    -------
    np.ndarray
        The exposure-adjusted grayscale image.
    """
    if len(img.shape) != 2:
        raise ValueError("Input image must be a 2D grayscale image.")
    
    image_norm = (img-np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

    image_clahe = exposure.equalize_adapthist(image_norm,
                                              clip_limit=0.01,
                                              kernel_size=(64,64)
                                              )
    
    return image_clahe

