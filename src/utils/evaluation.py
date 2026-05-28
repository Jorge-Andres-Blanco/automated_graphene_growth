from pathlib import Path
import os
from src.models import TransitionModel, EnsembleTransitionModel
from src.data_handling import TransitionDataLoader
from src.utils.plotting import plot_possible_actions_losses
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class Evaluator:
    """
    Evaluation suite for single and ensemble Latent Transition Models.
    Handles metric calculations, autoregressive rollouts, and plotting.
    """
    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device



    def evaluate_transition_model(self, model: TransitionModel, z_eval:np.ndarray, a_eval:np.ndarray, y_eval:np.ndarray):
        """
        Evaluates a trained TransitionModel on a given dataset, computing L2 distance,
        cosine similarity, and MSE loss between predicted and actual states. Also plots
        a PCA reduction of the trajectories and the step-wise metrics.
        
        Parameters
        ----------
        model : TransitionModel
            The trained TransitionModel to evaluate.
        z_eval : np.ndarray
            The input history of latent states. Shape: (T, history, 384).
        a_eval : np.ndarray
            The input history of action sequences. Shape: (T, history, action_dim).
        y_eval : np.ndarray
            The true target latent states. Shape: (T, 384).
            
        Returns
        -------
        l2_distances : torch.Tensor
            Step-wise L2 distances between predicted and actual states. Shape: (T,).
        cos_similarities : torch.Tensor
            Step-wise cosine similarities between predicted and actual states. Shape: (T,).
        mse_loss : float
            The mean squared error loss over the entire evaluation set.
            
        Hardcoded Elements
        ------------------
        - PCA components: Hardcoded to 2 for 2D visualization (`PCA(n_components=2)`).
        - Plot styling: Figure size (12, 5), colors, subplots arrangement.
        """

        model.to(self.device)
        model.eval()

        z_eval = torch.from_numpy(z_eval).float().to(self.device)
        a_eval = torch.from_numpy(a_eval).float().to(self.device)
        y_eval = torch.from_numpy(y_eval).float().to(self.device)

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



    def plot_trajectory_evaluation(self, y_pca, y_pred_pca, l2_distances, cos_similarities):

        """
        Plots the true vs predicted trajectory in a 2D PCA space, along with step-wise 
        L2 distance and cosine similarity metrics.
        
        Parameters
        ----------
        y_pca : np.ndarray
            The PCA-reduced true trajectory. Shape: (T, 2).
        y_pred_pca : np.ndarray
            The PCA-reduced predicted trajectory. Shape: (T, 2).
        l2_distances : np.ndarray or torch.Tensor
            Step-wise L2 distances. Shape: (T,).
        cos_similarities : np.ndarray or torch.Tensor
            Step-wise cosine similarities. Shape: (T,).
            
        Returns
        -------
        None
            Displays the generated matplotlib figure.
            
        Hardcoded Elements
        ------------------
        - Plot styling: Figure size (12, 5), scatter point sizes (s=20), alphas, 
        colors (blue, red, black), subplots (1x3).
        """

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


    def evaluate_ensemble_transition_model(self, ensemble_model: EnsembleTransitionModel, z_data: np.ndarray, a_data: np.ndarray, y_data: np.ndarray):
        """
        Evaluates a trained EnsembleTransitionModel on a dataset by computing metrics 
        between the ensemble mean predictions and actual states.
        
        Parameters
        ----------
        ensemble_model : EnsembleTransitionModel
            The trained ensemble model to evaluate.
        z_data : np.ndarray
            The input history of latent states. Shape: (T, history, 384).
        a_data : np.ndarray
            The input history of action sequences. Shape: (T, history, action_dim).
        y_data : np.ndarray
            The true target latent states. Shape: (T, 384).
            
        Returns
        -------
        l2_distances : torch.Tensor
            Step-wise L2 distances. Shape: (T,).
        cos_similarities : torch.Tensor
            Step-wise cosine similarities. Shape: (T,).
        mse_loss : float
            Mean squared error loss of the ensemble mean prediction.
            
        Hardcoded Elements
        ------------------
        - PCA components: Hardcoded to 2 for visualization.
        - Plot styling: Figure size (12, 5), colors, scatter settings.
        """
        ensemble_model.to(self.device)
        ensemble_model.eval()

        z_eval = torch.from_numpy(z_data).float().to(self.device)
        a_eval = torch.from_numpy(a_data).float().to(self.device)
        y_eval = torch.from_numpy(y_data).float().to(self.device)

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

    def evaluate_on_trajectory(self, model, z_traj, a_traj, y_traj):

        """
        Evaluates a TransitionModel autoregressively across an entire trajectory, 
        feeding its own predictions back as input for subsequent steps.
        
        Parameters
        ----------
        model : TransitionModel
            The trained TransitionModel to evaluate.
        z_traj : np.ndarray
            The sequence of true latent states. Expected shape: (T, history, 384).
        a_traj : np.ndarray
            The sequence of actions taken. Expected shape: (T, history, action_dim).
        y_traj : np.ndarray
            The sequence of target latent states. Expected shape: (T, 384).
            
        Returns
        -------
        pca_tuple : tuple of np.ndarray
            A tuple containing `(y_pca, y_pred_pca)`, which are the 2D PCA representations 
            of the true and predicted trajectories, respectively. Shapes: (T, 2), (T, 2).
        l2_distances : np.ndarray
            Step-wise L2 distances. Shape: (T,).
        cos_similarities : np.ndarray
            Step-wise cosine similarities. Shape: (T,).
            
        Hardcoded Elements
        ------------------
        - Latent dimension: Hardcoded to 384 for the allocation of `z_predicted_array`.
        - PCA components: Hardcoded to 2.
        """


        model.to(self.device)
        model.eval()

        z_traj = torch.from_numpy(z_traj).float().to(self.device)
        a_traj = torch.from_numpy(a_traj).float().to(self.device)
        y_traj = torch.from_numpy(y_traj).float().to(self.device) # shape (time,384)

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


    def evaluate_ensemble_on_trajectory(self, ensemble_model: EnsembleTransitionModel, z_traj: np.ndarray, a_traj: np.ndarray, y_traj: np.ndarray):

        """
        Evaluates an EnsembleTransitionModel autoregressively across an entire trajectory.
        At each step, it feeds the mean of the ensemble predictions back into the model 
        as the latent state for the next step.
        
        Parameters
        ----------
        ensemble_model : EnsembleTransitionModel
            The trained ensemble model to evaluate.
        z_traj : np.ndarray
            The sequence of true latent states. Expected shape: (T, history, 384).
        a_traj : np.ndarray
            The sequence of actions taken. Expected shape: (T, history, action_dim).
        y_traj : np.ndarray
            The sequence of target latent states. Expected shape: (T, 384).
            
        Returns
        -------
        pca_tuple : tuple of np.ndarray
            A tuple `(y_pca, y_pred_pca)` containing the 2D PCA representations of the 
            true and predicted trajectories. Shapes: (T, 2), (T, 2).
        l2_distances : np.ndarray
            Step-wise L2 distances. Shape: (T,).
        cos_similarities : np.ndarray
            Step-wise cosine similarities. Shape: (T,).
            
        Hardcoded Elements
        ------------------
        - Latent dimension: Hardcoded to 384 for the allocation of `z_predicted_array`.
        - PCA components: Hardcoded to 2.
        """


        ensemble_model.to(self.device)
        ensemble_model.eval()

        z_traj = torch.from_numpy(z_traj).float().to(self.device)
        a_traj = torch.from_numpy(a_traj).float().to(self.device)
        y_traj = torch.from_numpy(y_traj).float().to(self.device) # shape (time,384)

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


    def predict_next_steps(self, steps: int, model: TransitionModel, z_init: np.ndarray, a_init: np.ndarray, a_future: np.ndarray):
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


        model.to(self.device)
        model.eval()

        predictions = torch.zeros((steps, 384), device=self.device)
        is_ensemble = hasattr(model, 'num_models') or isinstance(model, EnsembleTransitionModel)
        
        if is_ensemble:
            std_predictions = torch.zeros((steps, 384), device=self.device)
        else:
            std_predictions = None

        # Ensure z_init has the correct 3D shape (1, history, 384)
        z_init_t = torch.from_numpy(z_init).float().to(self.device)
        if z_init_t.dim() == 1:
            z_init_t = z_init_t.unsqueeze(-1)
        z_hist = z_init_t.unsqueeze(0) 

        # Ensure a_init has the trailing action_dim (history, action_dim) -> (1, history, action_dim)
        a_init_t = torch.from_numpy(a_init).float().to(self.device)
        if a_init_t.dim() == 1:
            a_init_t = a_init_t.unsqueeze(-1) 
        a_hist = a_init_t.unsqueeze(0) 


        a_fut = torch.from_numpy(a_future).float().to(self.device)
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


    @staticmethod
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




    def evaluate_rollouts(self, steps: int, model, z_traj: np.ndarray, a_traj: np.ndarray, y_traj: np.ndarray):
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
            

        model.to(self.device)
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

        y_traj_t = torch.from_numpy(y_traj).float().to(self.device) # shape (time,384)

        with torch.no_grad():
        
            for i, t in enumerate(range(num_evals)):
                
                # z_traj and a_traj are ALREADY windowed from the dataloader
                z_init = z_traj[t] # Shape: (history, 384)
                a_init = a_traj[t] # Shape: (history, action_dim)
                
                # Extract future actions by taking the newest action from the subsequent overlapping windows
                a_future = np.array([a_traj[t + k][-1] for k in range(1, steps + 1)]) # Shape: (steps, action_dim)
                
                y_target = y_traj_t[t:t+steps] # Shape: (steps, 384)
                
                # prediction
                predictions, std_predictions = self.predict_next_steps(steps, model, z_init, a_init, a_future)
                
                dz[i] = self.extract_delta_z(z_init, predictions).cpu().numpy()
                
                if std_predictions is not None:
                    std_z[i] = std_predictions.cpu().numpy()
                    
                l2_distances[i] = torch.linalg.norm(predictions - y_target, dim=-1).cpu().numpy()
                cos_similarities[i] = torch.nn.functional.cosine_similarity(predictions, y_target, dim=-1).cpu().numpy()
                mse_losses[i] = torch.nn.functional.mse_loss(predictions, y_target, reduction='none').mean(dim=-1).cpu().numpy()

        return dz.mean(axis = 0), std_z.mean(axis = 0), l2_distances.mean(axis = 0), cos_similarities.mean(axis = 0)
    
    
    def analyze_and_plot_transition(self, model, data_processor, frame_0, frame_1, a0, save_path=None, actual_flow_sequence=None, frame_idx=None, target_idx=None):
        """
        Encodes images, analyzes a transition, predicts optimal actions, and saves the plot to disk.
        """
        model.to(self.device)
        model.eval()

        # Encode the frames internally
        embeddings = data_processor.encode_frames([frame_0, frame_1])
        z0, z1 = embeddings[0], embeddings[1]

        # 2. Calculate distances
        l2_distance = np.linalg.norm(z0 - z1)
        cosine_similarity = np.dot(z0, z1) / (np.linalg.norm(z0) * np.linalg.norm(z1))

        # 3. Predict action losses
        with torch.no_grad():
            losses, actions_evaluated = model.predict_action_losses(
                steps=5, 
                z_init=torch.tensor([z0], dtype=torch.float32).to(self.device), 
                a_init=torch.tensor([a0], dtype=torch.float32).to(self.device),
                a_pos="all", 
                target=torch.tensor(z1, dtype=torch.float32).to(self.device)
            )

        # 4. Create the Plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        axes[0, 0].imshow(frame_0, cmap='gray')
        axes[0, 0].set_title(f"Current State (Frame {frame_idx})" if frame_idx is not None else "Current State", fontsize=14)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(frame_1, cmap='gray')
        axes[0, 1].set_title(f"Target State (Frame {target_idx})" if target_idx is not None else "Target State", fontsize=14)
        axes[0, 1].axis('off')

        plot_possible_actions_losses(losses, actions_evaluated, aggregate='mean', ax=axes[1, 0])

        if actual_flow_sequence is not None and len(actual_flow_sequence) > 0:
            frames_range = np.arange(frame_idx, frame_idx + len(actual_flow_sequence))
            axes[1, 1].plot(frames_range, actual_flow_sequence, marker='o', color='orange', linewidth=2)
            axes[1, 1].set_ylim(0, max(np.max(actual_flow_sequence) * 1.1, 1.0))
        
        axes[1, 1].set_title("Actual Applied CH4 Flow", fontsize=14)
        axes[1, 1].set_xlabel("Frame Index", fontsize=12)
        axes[1, 1].set_ylabel("CH4 Flow (sccm)", fontsize=12)
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)

        title_text = (
            f"Transition Analysis\n"
            f"L2 Distance: {l2_distance:.4f}  |  Cosine Similarity: {cosine_similarity:.4f}"
        )
        plt.suptitle(title_text, fontsize=18, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
        else:
            plt.show()