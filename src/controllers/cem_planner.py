import torch
import numpy as np
from src.models.transition import EnsembleTransitionModel


class CEMPlanner:
    """
    Uses the World Model to evaluate potential future actions 
    and select the optimal path to a target state.
    """
    def __init__(self, transition_model: EnsembleTransitionModel, horizon:int=5):
        """
        Args:
            transition_model: The trained EnsembleTransitionModel.
            horizon (int): How many steps into the future to simulate.
            device (str): Compute device ('cpu' or 'cuda').
        """
        self.transition_model = transition_model
        self.horizon = horizon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ensure the model is in evaluation mode
        for model in self.transition_model.models:
            model.eval()

    def get_best_action(self, current_z, current_a, target_z, action_space="all"):
        """
        Evaluates potential actions and returns the best immediate CH4 flow to apply.
        
        Args:
            current_z (np.ndarray or Tensor): Current latent state embedding.
            current_a (float or Tensor): The currently applied CH4 flow.
            target_z (np.ndarray or Tensor): The goal latent state embedding.
            action_space (str): The subset of actions to evaluate (e.g., "all", "closer_7").
            
        Returns:
            float: The optimal CH4 flow value to execute next.
        """
        
        # 1. Format inputs for the model (ensure they are batches of sequences)
        if not isinstance(current_z, torch.Tensor):
            current_z = torch.from_numpy(current_z).float().unsqueeze(0).to(self.device)
        if not isinstance(current_a, torch.Tensor):
            current_a = torch.tensor([current_a], dtype=torch.float32).to(self.device)
        if not isinstance(target_z, torch.Tensor):
            target_z = torch.tensor(target_z, dtype=torch.float32).to(self.device)


        # 2. Ask the model to simulate the futures
        with torch.no_grad(): # Critical for speed: we are planning, not training!
            losses, actions_evaluated = self.transition_model.predict_action_losses(
                planning_horizon=self.horizon,
                z_init=current_z,
                a_init=current_a,
                a_pos=action_space,
                target=target_z
            )

        # 3. Process the losses (Aggregate across time steps and ensemble models)
        # losses shape: (num_actions, num_models, steps)
        losses_np = losses.cpu().numpy()
        actions_np = actions_evaluated.cpu().numpy()

        # Average loss across the temporal rollout (axis=2) and ensemble models (axis=1)
        mean_losses = losses_np.mean(axis=(1, 2))

        # Select the optimal action
        best_action_idx = np.argmin(mean_losses)
        best_action = actions_np[best_action_idx]

        print(f"[Planner] Best predicted action: {best_action:.2f} sccm (Loss: {mean_losses[best_action_idx]:.4f})")

        return float(best_action)
    

    def get_highest_variance_action(self, current_z, current_a, action_space="all"):
        """
        Evaluates potential actions and returns the CH4 flow that produces the highes variance in the predictions.
        
        Args:
            current_z (np.ndarray or Tensor): Current latent state embedding.
            current_a (float or Tensor): The currently applied CH4 flow.
            action_space (str): The subset of actions to evaluate (e.g., "all", "closer_7").
            
        Returns:
            float: The highest-variance CH4 flow to execute next.
        """
        
        # 1. Format inputs for the model (ensure they are batches of sequences)
        if not isinstance(current_z, torch.Tensor):
            current_z = torch.from_numpy(current_z).float().unsqueeze(0).to(self.device)
        if not isinstance(current_a, torch.Tensor):
            current_a = torch.tensor([current_a], dtype=torch.float32).to(self.device)


        # 2. Ask the model to simulate the futures
        # TODO

        with torch.no_grad(): # Critical for speed: we are planning, not training!
            predictions, actions_evaluated = self.transition_model.predict_action_results(
                planning_horizon=self.horizon,
                z_init=current_z,
                a_init=current_a,
                a_pos=action_space,
            )

        # 3. Process the predictions (Aggregate across time steps and ensemble models)
        # predictions shape: (num_actions, num_models, planning_horizon)
        predictions_np = predictions.cpu().numpy()
        actions_np = actions_evaluated.cpu().numpy()

        # Calculate std across the temporal (axis=2), ensemble models (axis=1)
        std_across_models_and_time = predictions_np.std(axis=(1, 2))
        
        # Average the uncertainty across all 384 latent features
        # Shape becomes (num_actions,) - exactly matching actions_np
        std_predictions = std_across_models_and_time.mean(axis=-1)

        # Select the optimal action
        best_action_idx = np.argmax(std_predictions)
        best_action = actions_np[best_action_idx]

        print(f"Action with highest predicted variance: {best_action}")

        return float(best_action)