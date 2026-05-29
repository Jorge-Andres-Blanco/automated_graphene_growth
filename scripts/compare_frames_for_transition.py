from pathlib import Path
import h5py
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from src.data_handling.hdf5_processor import HDF5Processor
from src.models import DinoEncoder, EnsembleTransitionModel
from src.utils.plotting import compare_images_in_latent_space, plot_possible_actions_losses
from src.utils.evaluation import Evaluator



train_data_path = Path("/data/lmcat/Computer_vision/training_data")
validation_data_path = Path("/data/lmcat/Computer_vision/validation_data")


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hist = 1
    step_size = 30
    hidden_dimension=1024
    normalization="layer"
    activation="leaky_relu"
    ensemble_model = ensemble_model = EnsembleTransitionModel(num_models=5,
                                                        latent_dim=384,
                                                        action_dim=1,
                                                        hidden_dim=1024,
                                                        normalization=normalization,
                                                        activation=activation,
                                                        num_hidden_layers=2,
                                                        history=hist)

    model_name_prefix = f"/data/lmcat/Computer_vision/models/mlp_activation_{activation}_norm_{normalization}_hist{hist}_step{step_size}_hiddim{hidden_dimension}"

    
    # Training/calling the model
    try:
        for i, model in enumerate(ensemble_model.models):
            model.load_state_dict(torch.load(f"{model_name_prefix}_transition_model_{i}.pth", map_location=device))
            print(f"model {i} loaded")
    except FileNotFoundError:
            print(f"Model {i} not found")

    data_processor = HDF5Processor(encoder=DinoEncoder())

    steps_in_future = 2
    movie_num = 2
    initial_frame_idx = 2000

    target_frame_idx = initial_frame_idx + step_size*steps_in_future
    frame_0 = data_processor.get_frame_data(movie_num, initial_frame_idx)
    frame_1 = data_processor.get_frame_data(movie_num, target_frame_idx)
    a0 = data_processor.get_frame_data(movie_num, initial_frame_idx, measurement="CH4")
    actual_flow = data_processor.get_frame_data(movie_num, np.arange(initial_frame_idx, target_frame_idx+1), measurement="CH4").flatten()

    # Encoding
    evaluator = Evaluator(device=device)
    save_path = f"/data/lmcat/Computer_vision/automated_graphene_growth/plots/transition_comparison_movie{movie_num}_frame{initial_frame_idx}_to_frame{target_frame_idx}.png"
    evaluator.analyze_and_plot_transition(ensemble_model,
                                            data_processor=HDF5Processor(encoder=DinoEncoder()),
                                            frame_0=frame_0, frame_1=frame_1, a0=a0,
                                            actual_flow_sequence=actual_flow, save_path=save_path,
                                            frame_idx=initial_frame_idx, target_idx=target_frame_idx)

    return None


if __name__ == "__main__":

    main()