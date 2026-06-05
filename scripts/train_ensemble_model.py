from pathlib import Path
import torch
import numpy as np
from src.data_handling import TransitionDataLoader
from src.utils.evaluation import Evaluator
from src.models import EnsembleTransitionModel, Trainer
import argparse

if __name__ == "__main__":

    # Data paths
    train_data_path = Path("/data/lmcat/Computer_vision/training_data")
    validation_data_path = Path("/data/lmcat/Computer_vision/validation_data")


    #Define hyperparameters
    activation = "leaky_relu"
    hidden_dimension = 1024
    hist = 1
    step_size = 30
    normalization = "layer"

    #Define model
    model_name_prefix = f"/data/lmcat/Computer_vision/models/mlp_activation_{activation}_norm_{normalization}_hist{hist}_step{step_size}_hiddim{hidden_dimension}"
    ensemble_model = EnsembleTransitionModel(num_models=8,
                                                    latent_dim=384,
                                                    action_dim=1,
                                                    hidden_dim=hidden_dimension,
                                                    normalization=normalization,
                                                    activation=activation,
                                                    num_hidden_layers=2,
                                                    history=hist)

    # To be changed according to the executing machine
    trainer = Trainer(lr=1e-3, batch_size=64, epochs=5)
    train_data_loader = TransitionDataLoader(train_data_path, step_size=step_size, hist_length=hist)
    
    ensemble_model = trainer.train_ensemble_with_bagging(ensemble_model=ensemble_model,
                                                            data_loader = train_data_loader,
                                                            save_prefix = model_name_prefix)
    
    trainer.plot_training_loss_vs_epoch()
    
    
    losses = trainer.losses
    losses_mean = np.mean(losses, axis=0)
    losses_std = np.std(losses, axis=0)
    print(f"Ensemble training completed. Last loss mean and std: {losses_mean[-1]}, {losses_std[-1]}")