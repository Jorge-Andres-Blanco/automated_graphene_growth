"""
Tasks: Building a World-Model control loop.

This file walks you through the pipeline for controlling a system via a
learned world model. The architecture is:

    Observer  →  World Model  →  Setter
       ↑                          |
       └──────── Environment ─────┘

  - Observer:    Reads the current state from the environment.
                 Returns the latent representation z_t and current controls a_t.
                 (In a real experiment: camera frame → DINOv2 → CLS token)

  - World Model: Predicts the next latent state given current state and action.
                 z_{t+1} = f(z_t, a_t)
                 (Learned from collected trajectory data)

  - Setter:      Applies an action to the environment.
                 (In a real experiment: sends new control values to the instrument)

"""

import torch
import torch.nn as nn


class TransitionModel(nn.Module):
    """
    Simple MLP transition model: z_{t+1} = f(z_(t-h):t, a_(t-h):t)

    """
    
    def __init__(self, latent_dim=384, action_dim=1, hidden_dim=512, num_hidden_layers=2, history = 5):
        """
        How to use:
        
        model = TransitionModel()

        pred_z = model(z[t-h:t], a[t-h:t_t)

        That is, with history h:

        pred_z = model(z_(t-h), z_(t-h+1), ..., z_t, a_(t-h), a_(t-h+1), ..., a_t)
        
        """

        super().__init__()

        self.history = history
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers


        input_dim = latent_dim*history + action_dim*history


        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]

        for _ in range(num_hidden_layers):
        
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(p=0.15)]
        
        layers.append(nn.Linear(hidden_dim, latent_dim))

        self.net = nn.Sequential(*layers)

    
    def forward(self, z_hist, a_hist):
        """
        Parameters
        ----------
            z_hist : torch.Tensor with shape (batch_size, history, latent_dim)
            a_hist : torch.Tensor with shape (batch_size, history, action_dim), action_dim is 1 for the moment
            
        Returns
        -------
            pred_z : torch.Tensor with shape (batch_size, latent_dim)
              
        """

        batch_size = z_hist.shape[0]

        z_hist_flat = z_hist.reshape(batch_size, -1)
        a_hist_flat = a_hist.reshape(batch_size, -1)

        x = torch.cat([z_hist_flat, a_hist_flat], dim=1)

        pred_delta_z = self.net(x)

        pred_z = z_hist[:,-1,:] + pred_delta_z

        return pred_z
    
    @torch.no_grad()
    def predict_next_steps(self, steps: int, z_init: torch.Tensor, a_init: torch.Tensor, a_fut: torch.Tensor):
        """
        Generates open-loop autoregressive predictions for state transitions over a given number of future steps.

        Parameters
        ----------
        steps : int
            The number of future steps to predict.
        z_init : torch.Tensor
            The initial latent state history. Expected shape: (history, latent_dim) or (latent_dim,).
        a_init : torch.Tensor
            The initial action history that led to `z_init`. Expected shape: (history,) or (history, action_dim).
        a_fut : torch.Tensor
            The intended future actions for the next `steps`. Expected shape: (steps,) or (steps, action_dim).

        Returns
        -------
        predictions : torch.Tensor
            The predicted mean latent states for the next `steps`. Shape: (steps, latent_dim).

        Hardcoded Elements
        ------------------
        - Latent dimension: Hardcoded to 384 during the pre-allocation of the `predictions` and `std_predictions` tensors (e.g., `torch.zeros((steps, 384))`).
        """

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.to(device)
        self.eval()

        predictions = torch.zeros((steps, 384), device=device)

        # Handle z_init dimensions: ensure (batch, history, latent_dim)
        if z_init.dim() == 1:
            z_init = z_init.unsqueeze(-1) # (latent_dim, 1) - though latent_dim is usually the last dim
        z_hist = z_init if z_init.dim() == 3 else z_init.unsqueeze(0) # (1, history, latent_dim)


        # Ensure a_init is also (1, history, action_dim)
        if a_init.dim() == 1:
            a_init = a_init.unsqueeze(-1) 
        a_hist = a_init if a_init.dim() == 3 else a_init.unsqueeze(0)

        # Ensure a_fut is (steps, action_dim)
        if a_fut.dim() == 1:
            a_fut = a_fut.unsqueeze(-1)

        for t in range(steps):

            z_next = self(z_hist, a_hist)
            predictions[t] = z_next.squeeze(0)

            # Update sliding windows by dropping oldest and appending newest
            z_hist = torch.cat((z_hist[:, 1:, :], z_next.unsqueeze(0)), dim=1)
            a_hist = torch.cat((a_hist[:, 1:, :], a_fut[t].unsqueeze(0).unsqueeze(0)), dim=1)

        return predictions




class EnsembleTransitionModel(nn.Module):

    def __init__(self, num_models: int, **kwargs):
        super().__init__()        
        self.num_models = num_models

        self.models = nn.ModuleList([TransitionModel(**kwargs) for _ in range(num_models)])

    def forward(self, z_hist, a_hist):

        """
        Parameters
        ----------
            z_hist : torch.Tensor with shape (batch_size, history, latent_dim)
            a_hist : torch.Tensor with shape (batch_size, history, action_dim), action_dim is 1 for the moment (Only CH4)
            
        Returns
        -------
            pred_z : torch.Tensor with shape (num_models, batch_size, latent_dim)

        """

        pred_z = torch.stack([model(z_hist, a_hist) for model in self.models], dim=0)

        return pred_z
    
    @torch.no_grad()
    def get_stats(self, z_hist: torch.Tensor, a_hist: torch.Tensor):
        """
        Get mean and std of the ensemble predictions.

        Parameters
        ----------
            z_hist : torch.Tensor with shape (batch_size, history, latent_dim)
            a_hist : torch.Tensor with shape (batch_size, history, action_dim), action_dim is 1 for the moment (Only CH4)
            
        Returns
        -------
            mean_pred_z : torch.Tensor with shape (batch_size, latent_dim)
            std_pred_z : torch.Tensor with shape (batch_size, latent_dim)

        """

        pred_z = self.forward(z_hist, a_hist)

        mean_pred_z = pred_z.mean(dim=0)
        std_pred_z = pred_z.std(dim=0)

        return mean_pred_z, std_pred_z
    

    @torch.no_grad()
    def predict_next_steps(self, z_init: torch.Tensor, a_init: torch.Tensor, a_fut: torch.Tensor):
        """
        Generates open-loop autoregressive predictions for state transitions over future steps
        for each model in the ensemble.

        Parameters
        ----------
        z_init : torch.Tensor
            The initial latent state history. Expected shape: (batch_size, history, latent_dim) or (history, latent_dim).
        a_init : torch.Tensor
            The initial action history that led to `z_init`.
        a_fut : torch.Tensor
            The intended future actions. The number of steps to predict is inferred from its length.
            
        Returns
        -------
        predictions : torch.Tensor
            The predicted latent states from all ensemble models. Shape: (num_models, steps, 384).
        """
        steps = a_fut.shape[0]
        predictions = torch.zeros((self.num_models, steps, 384), device=z_init.device)

        for i, model in enumerate(self.models):

            predictions_model = model.predict_next_steps(steps, z_init, a_init, a_fut)
            predictions[i] = predictions_model

        return predictions
    


    @torch.no_grad()
    def predict_action_results(self, steps: int, z_init: torch.Tensor, a_init: torch.Tensor, a_pos:str = "all", target:torch.Tensor = None):
        """
        Evaluates hypothetical constant-action sequences over a future horizon and 
        scores them against a target state.
        """

        device = z_init.device

        # Default target: maintain the current state
        if target is None:
            target = z_init[:, -1, :] # Shape: (batch_size, 384) or (384,)

        # Define Action Space
        if a_pos == "all":
            a_search = torch.arange(0, 20, 1)
        elif a_pos == "closer_5":
            a_current = int(a_init[0, -1, 0])
            a_search = torch.arange(a_current - 2, a_current + 3, 1)
        elif a_pos == "closer_7":
            a_current = int(a_init[0, -1, 0])
            a_search = torch.arange(a_current - 3, a_current + 4, 1)
        else:
            raise ValueError("Invalid a_pos. Use 'all', 'closer_5', or 'closer_7'.")

        # Clamp actions to ensure they don't fall below zero
        a_search = torch.clamp(a_search, min=0)

        possibilities = len(a_search)

        # Predictions
        predictions = torch.zeros((possibilities, self.num_models, steps, 384), device=device)
        
        for i, a_fut in enumerate(a_search):
            
            a_fut = a_fut * torch.ones(steps, 1, device=device)

            predictions[i] = self.predict_next_steps(z_init, a_init, a_fut)


        # Target expansion to shape (1,1,1,384) and then as predicions (possibilities, num_models, steps, 384)
        target_exp = target.view(1,1,1,-1).expand_as(predictions)

        # Loss
        mse_loss = nn.functional.mse_loss(predictions, target_exp, reduction='none').mean(dim=-1)
        cos_sim = nn.functional.cosine_similarity(predictions, target_exp, dim=-1)

        alpha = 1.0
        combined_loss = mse_loss + alpha*(1-cos_sim)

        # Average across all models (dim 1)
        losses = combined_loss.mean(dim=1)

        return losses #shape (possibilities, steps)