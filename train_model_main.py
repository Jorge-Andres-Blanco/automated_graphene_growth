from pathlib import Path
import torch
from data_processing.data_loader import load_transition_data
from WM_JABV.train_transition_model import train_transition_model
import WM_JABV.evaluation as eval
from WM_JABV.transition_model import TransitionModel


def main():

    # To be changed according to the executing machine
    train_data_path = Path(r"\\dfs\data\lmcat\Computer_vision\training_data")
    validation_data_path = Path(r"\\dfs\data\lmcat\Computer_vision\validation_data")


    hist = 5
    step_size = 5
    train = False
    model = TransitionModel()

    
    if train:
        
        z_train, a_train, y_train = load_transition_data(train_data_path, step_size = step_size, hist_length = hist)
        model = train_transition_model(z_train, a_train, y_train, model=model, epochs=250, lr=1e-3, batch_size=64, save_model_as = "transition_model.pth")
    
    else:
    
        model.load_state_dict(torch.load("transition_model.pth"))


    z_eval, a_eval, y_eval = load_transition_data(validation_data_path, step_size = step_size, hist_length = hist)

    eval.evaluate_transition_model(model, z_eval, a_eval, y_eval)

    # Validation trajectory 1
    (y_pca, y_pred_pca), l2_distances, cos_similarities = eval.evaluate_on_trajectory(model, z_eval[:48], a_eval[:48], y_eval[:48])
    eval.plot_trajectory_evaluation(y_pca, y_pred_pca, l2_distances, cos_similarities)

    # Validation trajectory 2
    (y_pca, y_pred_pca), l2_distances, cos_similarities = eval.evaluate_on_trajectory(model, z_eval[48:78], a_eval[48:78], y_eval[48:78])
    eval.plot_trajectory_evaluation(y_pca, y_pred_pca, l2_distances, cos_similarities)

    # Validation trajectory 3
    (y_pca, y_pred_pca), l2_distances, cos_similarities = eval.evaluate_on_trajectory(model, z_eval[78:108], a_eval[78:108], y_eval[78:108])
    eval.plot_trajectory_evaluation(y_pca, y_pred_pca, l2_distances, cos_similarities)

    #Validation trajectory 4
    (y_pca, y_pred_pca), l2_distances, cos_similarities = eval.evaluate_on_trajectory(model, z_eval[108:136], a_eval[108:136], y_eval[108:136])
    eval.plot_trajectory_evaluation(y_pca, y_pred_pca, l2_distances, cos_similarities)

    #Validation trajectory 5
    (y_pca, y_pred_pca), l2_distances, cos_similarities = eval.evaluate_on_trajectory(model, z_eval[136:166], a_eval[136:166], y_eval[136:166])
    eval.plot_trajectory_evaluation(y_pca, y_pred_pca, l2_distances, cos_similarities)

    #Validation trajectory 6
    (y_pca, y_pred_pca), l2_distances, cos_similarities = eval.evaluate_on_trajectory(model, z_eval[166:206], a_eval[166:206], y_eval[166:206])
    eval.plot_trajectory_evaluation(y_pca, y_pred_pca, l2_distances, cos_similarities)

    #Validation trajectory 7
    (y_pca, y_pred_pca), l2_distances, cos_similarities = eval.evaluate_on_trajectory(model, z_eval[206:236], a_eval[206:236], y_eval[206:236])
    eval.plot_trajectory_evaluation(y_pca, y_pred_pca, l2_distances, cos_similarities)

    return None


if __name__ == "__main__":

    main()