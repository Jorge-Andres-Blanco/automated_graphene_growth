from pathlib import Path
import torch
from data_processing.data_loader import load_transition_data
from WM_JABV.train_transition_model import train_transition_model, plot_training_loss
import WM_JABV.evaluation as eval
from WM_JABV.transition_model import TransitionModel


def main():

    # To be changed according to the executing machine
    train_data_path = Path(r"\\dfs\data\lmcat\Computer_vision\training_data")
    validation_data_path = Path(r"\\dfs\data\lmcat\Computer_vision\validation_data")


    hist = 13
    step_size = 7
    train = True
    model = TransitionModel(history=hist)

    
    if train:
        
        z_train, a_train, y_train = load_transition_data(train_data_path, step_size = step_size, hist_length = hist)
        model, losses = train_transition_model(z_train, a_train, y_train, model=model, epochs=50, lr=1e-3, batch_size=64, save_model_as = "transition_model.pth")
        plot_training_loss(losses)
    
    else:
    
        model.load_state_dict(torch.load("transition_model.pth"))


    z_eval, a_eval, y_eval, indices = load_transition_data(validation_data_path, step_size = step_size, hist_length = hist, return_indices=True)
    print (z_eval.shape, a_eval.shape, y_eval.shape)

    l2_distances, cos_similarities, mse_loss = eval.evaluate_transition_model(model, z_eval, a_eval, y_eval)

    print(f"MSE Loss on validation data: {mse_loss}")

    print(indices)

    for (i, f) in indices:
        if (f-i) < hist+1:
            print(f"Skipping evaluation for indices {i} to {f} due to insufficient length.")
            continue
        (y_pca, y_pred_pca), l2_distances, cos_similarities = eval.evaluate_on_trajectory(model, z_eval[i:f], a_eval[i:f], y_eval[i:f])
        eval.plot_trajectory_evaluation(y_pca, y_pred_pca, l2_distances, cos_similarities)

    return None


if __name__ == "__main__":

    main()