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
