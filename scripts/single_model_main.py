from pathlib import Path
import torch
import numpy as np
from src.data_handling import TransitionDataLoader
from src.models import Trainer, TransitionModel
from src.utils.evaluation import Evaluator


def main():

    # To be changed according to the executing machine
    train_data_path = Path(r"\\dfs\data\lmcat\Computer_vision\training_data")
    validation_data_path = Path(r"\\dfs\data\lmcat\Computer_vision\validation_data")


    hist = 15
    step_size = 5
    train = True
    model = TransitionModel(history=hist)
    trainer = Trainer()
    evalua = Evaluator()

    train_data_loader = TransitionDataLoader(train_data_path, step_size=step_size, hist_length=hist)
    validation_data_loader = TransitionDataLoader(validation_data_path, step_size=step_size, hist_length=hist)

    # Training 
    
    if train:
        
        z_train, a_train, y_train = train_data_loader.load_full_dataset()
        model, losses = trainer.train_transition_model(z_train, a_train, y_train, model=model, epochs=50, lr=1e-3, batch_size=64, save_model_as = "transition_model.pth")
        trainer.plot_training_loss_vs_epoch()
    
    else:
    
        model.load_state_dict(torch.load("transition_model.pth"))


    z_eval, a_eval, y_eval, indices = validation_data_loader.load_full_dataset(return_indices=True)

    l2_distances, cos_similarities, mse_loss = evalua.evaluate_transition_model(model, z_eval, a_eval, y_eval)

    print(f"MSE Loss on validation data: {mse_loss}")

    print(indices)

    for (i, f) in indices:
        if (f-i) < hist+1:
            print(f"Skipping evaluation for indices {i} to {f} due to insufficient length.")
            continue
        (y_pca, y_pred_pca), l2_distances, cos_similarities = evalua.evaluate_on_trajectory(model, z_eval[i:f], a_eval[i:f], y_eval[i:f])
        evalua.plot_trajectory_evaluation(y_pca, y_pred_pca, l2_distances, cos_similarities)

    return None


if __name__ == "__main__":

    main()