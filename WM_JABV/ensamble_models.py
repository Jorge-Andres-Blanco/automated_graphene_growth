import torch
import torch.nn as nn
from WM_JABV.transition_model import TransitionModel


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
    def get_stats(self, z_hist, a_hist):
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