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

        self.register_buffer('a_min', torch.tensor(0.0))
        self.register_buffer('a_max', torch.tensor(15.0))


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
        actions_clamped = torch.clamp(a_hist, min=self.a_min, max=self.a_max)

        a_scaled = (2 * (actions_clamped - self.a_min) / (self.a_max - self.a_min)) - 1.0 # Maps action values from [-1,1]

        # Flatten
        batch_size = z_hist.shape[0]

        z_hist_flat = z_hist.reshape(batch_size, -1)
        a_hist_flat = a_scaled.reshape(batch_size, -1)

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

        # Hanfle dimensions
        z_hist = z_init.view(1, self.history, self.latent_dim)
        a_hist = a_init.view(1, self.history, self.action_dim)
        a_fut_seq = a_fut.view(steps, self.action_dim)

        for t in range(steps):

            z_next = self(z_hist, a_hist)
            predictions[t] = z_next.squeeze(0)

            # z_next.unsqueeze(1) transforms (1, 384) -> (1, 1, 384)
            z_hist = torch.cat((z_hist[:, 1:, :], z_next.unsqueeze(1)), dim=1)

            # Update history by dropping oldest and appending newest
            z_hist = torch.cat((z_hist[:, 1:, :], z_next.unsqueeze(0)), dim=1)
            # Extract the specific action and shape it to (1, 1, action_dim)
            next_a = a_fut_seq[t].view(1, 1, self.action_dim)
            a_hist = torch.cat((a_hist[:, 1:, :], next_a), dim=1)

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
    def predict_action_losses(self, steps: int, z_init: torch.Tensor, a_init: torch.Tensor, a_pos:str = "all", target:torch.Tensor = None):
        """
        Evaluates hypothetical constant-action sequences over a future horizon and 
        scores them against a target state.
        """

        device = z_init.device

        # Default target: maintain the current state
        if target is None:
            target = z_init.view(-1, 384)[-1:]

        # Define Action Space
        # Every 0.5 to analyze intermediate values
        if a_pos == "all":
            a_search = torch.arange(0, 20, 0.5, device=device)
        elif a_pos == "closer_5":
            a_current = a_init[-1].item()
            a_search = torch.arange(a_current - 2, a_current + 3, 0.5, device=device)
        elif a_pos == "closer_7":
            a_current = a_init[-1].item()
            a_search = torch.arange(a_current - 3, a_current + 4, 0.5, device=device)
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

        # Returns combined_loss (possibilities, num_models) and a_search (possibilities,)
        return combined_loss, a_search
    

    @torch.no_grad()
    def predict_next_step(self, steps: int, z_init: torch.Tensor, a_init: torch.Tensor, a_pos:str = "all", target:torch.Tensor = None, aggregate_steps:str = "horizon_loss"):

        """
        Returns best next step with standard deviation
        """
        device = z_init.device
        losses, a_search = self.predict_action_losses(steps, z_init, a_init, a_pos, target)
        
        if aggregate_steps == "horizon_loss":
            losses = losses[:,:,-1]
        elif aggregate_steps == "mean":
            losses = losses.mean(dim=-1)
        elif aggregate_steps == "cumulative_sum":
            losses = losses.sum(dim=-1)
        else:
            raise ValueError("Invalid aggregate_steps. Use 'horizon_loss', 'mean', or 'cumulative_sum'.")
        

        #Get the best_action of each model

        best_action_indices = losses.argmin(dim=0) #Shape (num_models,)

        best_actions = a_search[best_action_indices] #Shape (num_models,)

        next_action_pred = best_actions.mean().item()
        next_action_std = best_actions.std().item()

        return next_action_pred, next_action_std
