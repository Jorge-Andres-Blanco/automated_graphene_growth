from WM_JABV.ensamble_models import EnsembleTransitionModel
from WM_JABV.transition_model import TransitionModel
from WM_JABV.train_transition_model import train_transition_model

def train_ensemble_transition_model(ensemble_model: EnsembleTransitionModel, z_data, a_data, y_data, **kwargs):

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

    for i, model in enumerate(ensemble_model.models):
        print(f"Training model {i+1}/{ensemble_model.num_models}")
        train_transition_model(z_data, a_data, y_data, model=model, save_model_as = f"transition_model_{i}.pth", **kwargs)
        
    return ensemble_model