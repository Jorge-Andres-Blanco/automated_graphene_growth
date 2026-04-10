from WM_JABV.transition_models import TransitionModel, EnsembleTransitionModel
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def evaluate_transition_model(model: TransitionModel, z_eval:np.ndarray, a_eval:np.ndarray, y_eval:np.ndarray):
    """
    Parameters:
        model: trained TransitionModel
        z_eval: np.ndarray (T, 384) — latent states
        a_eval: np.ndarray (T, 1) — action sequence
        y_eval: np.ndarray (T, 384) — observed states
    """

    #   # Compare y_pred vs y
    #   # Plot with PCA or UMAP
    #   # Compute per-step cosine similarity
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    z_eval = torch.from_numpy(z_eval).float().to(device)
    a_eval = torch.from_numpy(a_eval).float().to(device)
    y_eval = torch.from_numpy(y_eval).float().to(device)

    with torch.no_grad():

        y_pred = model(z_eval, a_eval)

        l2_distances = torch.linalg.norm(y_pred - y_eval, dim = -1)
        cos_similarities = nn.functional.cosine_similarity(y_pred, y_eval, dim=-1)
        mse_loss = nn.MSELoss()(y_pred, y_eval)

        # PCA

        pca = PCA(n_components=2)
        y_pca = pca.fit_transform(y_eval)
        y_pred_pca = pca.transform(y_pred)

    # Plot

    # PCA: one trajectory as a line
    plt.figure(figsize=(12, 5)) 
    plt.subplot(1, 3, 1)

    plt.scatter(y_pca[:, 0], y_pca[:, 1], color="blue", alpha=0.65, label="Real")
    plt.scatter(y_pred_pca[:, 0], y_pred_pca[:, 1], color="red", alpha=0.5, label="Predicted")
        
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(l2_distances, label='L2 Distance', color = 'red')
    plt.xlabel('Measurement')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(cos_similarities, label='Cosine Similarity')
    plt.xlabel('Measurement')
    plt.legend()

    plt.suptitle('Transition Model Evaluation')
    plt.show()

    return l2_distances, cos_similarities, mse_loss.item()


def evaluate_on_trajectory(model, z_traj, a_traj, y_traj):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    z_traj = torch.from_numpy(z_traj).float().to(device)
    a_traj = torch.from_numpy(a_traj).float().to(device)
    y_traj = torch.from_numpy(y_traj).float().to(device) # shape (time,384)

    time = z_traj.shape[0]
    l2_distances = np.zeros(time)
    cos_similarities = np.zeros(time)
    z_predicted_array = np.zeros((time, 384))
    
    h = model.history
    
    z = z_traj[0].unsqueeze(0) # Shape (1,hist,384)

    with torch.no_grad():
    
        for t in range(time):

            z_pred = model(z, a_traj[t].unsqueeze(0)) #shape (1,384)

            z = torch.cat((z[:,1:], z_pred.unsqueeze(0)), dim=1)

            z_predicted_array[t] = z_pred.squeeze(0).cpu().numpy()

            l2_distances[t] = torch.linalg.norm(z_pred-y_traj[t], dim = -1)

            cos_similarities[t] = nn.functional.cosine_similarity(z_pred, y_traj[t], dim=-1)

    pca = PCA(n_components=2)
    y_pca = pca.fit_transform(y_traj.cpu().numpy())
    y_pred_pca = pca.transform(z_predicted_array)

    return (y_pca, y_pred_pca), l2_distances, cos_similarities



def plot_trajectory_evaluation(y_pca, y_pred_pca, l2_distances, cos_similarities):

    # PCA: one trajectory as a line
    plt.figure(figsize=(12, 5)) 
    plt.subplot(1, 3, 1)

    plt.plot(y_pca[:, 0], y_pca[:, 1], color="blue", label="Real")
    plt.scatter(y_pca[:, 0], y_pca[:, 1], color="blue", alpha = 0.4, s = 20)
    
    plt.plot(y_pred_pca[:, 0], y_pred_pca[:, 1], color="red", label="Predicted")
    plt.scatter(y_pred_pca[:, 0], y_pred_pca[:, 1], color="red", alpha = 0.4, s=20)

    plt.scatter(y_pca[0, 0], y_pca[0, 1], color='black', s=20, label = "Start")
    plt.scatter(y_pred_pca[0, 0], y_pred_pca[0, 1], color='black', s=20)
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(l2_distances, label='L2 Distance', color = 'red')
    plt.xlabel('Steps')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(cos_similarities, label='Cosine Similarity')
    plt.xlabel('Steps')
    plt.legend()

    plt.suptitle('Transition Model Evaluation in a trajectory')
    plt.show()


def evaluate_ensemble_transition_model(ensemble_model: EnsembleTransitionModel, z_data: np.ndarray, a_data: np.ndarray, y_data: np.ndarray):

    """
    Parameters:
        ensemble_model: trained EnsembleTransitionModel
        z_data: np.ndarray (T, 384) — latent states
        a_data: np.ndarray (T, 2) — action sequence
        y_data: np.ndarray (T, 384) — observed states
    """

    #   # Compare y_pred vs y
    #   # Plot with PCA or UMAP
    #   # Compute per-step cosine similarity
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ensemble_model.to(device)
    ensemble_model.eval()

    z_eval = torch.from_numpy(z_data).float().to(device)
    a_eval = torch.from_numpy(a_data).float().to(device)
    y_eval = torch.from_numpy(y_data).float().to(device)

    with torch.no_grad():

        mean_pred, std_pred = ensemble_model.get_stats(z_eval, a_eval)

        l2_distances = torch.linalg.norm(mean_pred - y_eval, dim = -1)
        cos_similarities = nn.functional.cosine_similarity(mean_pred, y_eval, dim=-1)
        mse_loss = nn.MSELoss()(mean_pred, y_eval)

        # PCA

        pca = PCA(n_components=2)
        y_pca = pca.fit_transform(y_eval)
        y_pred_pca = pca.transform(mean_pred)

    # Plot

    # PCA: one trajectory as a line
    plt.figure(figsize=(12, 5)) 
    plt.subplot(1, 3, 1)

    plt.scatter(y_pca[:, 0], y_pca[:, 1], color="blue", alpha=0.65, label="Real")
    plt.scatter(y_pred_pca[:, 0], y_pred_pca[:, 1], color="red", alpha=0.5, label="Predicted")
        
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(l2_distances, label='L2 Distance', color = 'red')
    plt.xlabel('Measurement')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(cos_similarities, label='Cosine Similarity')
    plt.xlabel('Measurement')
    plt.legend()

    plt.suptitle('Transition Model Evaluation')
    plt.show()

    return l2_distances, cos_similarities, mse_loss.item()

def evaluate_on_trajectory(model, z_traj, a_traj, y_traj):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    z_traj = torch.from_numpy(z_traj).float().to(device)
    a_traj = torch.from_numpy(a_traj).float().to(device)
    y_traj = torch.from_numpy(y_traj).float().to(device) # shape (time,384)

    time = z_traj.shape[0]
    l2_distances = np.zeros(time)
    cos_similarities = np.zeros(time)
    z_predicted_array = np.zeros((time, 384))
    
    h = model.history
    
    z = z_traj[0].unsqueeze(0) # Shape (1,hist,384)

    with torch.no_grad():
    
        for t in range(time):

            z_pred = model(z, a_traj[t].unsqueeze(0)) #shape (1,384)

            z = torch.cat((z[:,1:], z_pred.unsqueeze(0)), dim=1)

            z_predicted_array[t] = z_pred.squeeze(0).cpu().numpy()

            l2_distances[t] = torch.linalg.norm(z_pred-y_traj[t], dim = -1)

            cos_similarities[t] = nn.functional.cosine_similarity(z_pred, y_traj[t], dim=-1)

    pca = PCA(n_components=2)
    y_pca = pca.fit_transform(y_traj.cpu().numpy())
    y_pred_pca = pca.transform(z_predicted_array)

    return (y_pca, y_pred_pca), l2_distances, cos_similarities


def evaluate_ensemble_on_trajectory(ensemble_model: EnsembleTransitionModel, z_traj: np.ndarray, a_traj: np.ndarray, y_traj: np.ndarray):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ensemble_model.to(device)
    ensemble_model.eval()

    z_traj = torch.from_numpy(z_traj).float().to(device)
    a_traj = torch.from_numpy(a_traj).float().to(device)
    y_traj = torch.from_numpy(y_traj).float().to(device) # shape (time,384)

    time = z_traj.shape[0]
    l2_distances = np.zeros(time)
    cos_similarities = np.zeros(time)
    z_predicted_array = np.zeros((time, 384))
    
    z = z_traj[0].unsqueeze(0) # Shape (1,hist,384)

    with torch.no_grad():
    
        for t in range(time):

            z_pred, std_pred = ensemble_model.get_stats(z, a_traj[t].unsqueeze(0)) #shape (num_models, 1,384)

            z_mean = z_pred.mean(dim=0, keepdim=True) #shape (1,384)

            z = torch.cat((z[:,1:], z_mean.unsqueeze(0)), dim=1)

            z_predicted_array[t] = z_mean.squeeze(0).cpu().numpy()

            l2_distances[t] = torch.linalg.norm(z_mean-y_traj[t], dim = -1)

            cos_similarities[t] = nn.functional.cosine_similarity(z_mean, y_traj[t], dim=-1)

    pca = PCA(n_components=2)
    y_pca = pca.fit_transform(y_traj.cpu().numpy())
    y_pred_pca = pca.transform(z_predicted_array)

    return (y_pca, y_pred_pca), l2_distances, cos_similarities


def predict_next_steps(steps, model, z_init, a_seq):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    if isinstance(model, EnsembleTransitionModel):
        predictions = torch.zeros((model.num_models, steps, 384)).to(device)
    else:
        predictions = torch.zeros((steps, 384)).to(device)

    z_init = torch.from_numpy(z_init).float().to(device)
    a_seq = torch.from_numpy(a_seq).float().to(device)

    with torch.no_grad():

        z = z_init.unsqueeze(0) # Shape (1,hist,384)

        for t in range(steps):

            if isinstance(model, EnsembleTransitionModel):
                z_pred, std_pred  = model.get_stats(z, a_seq[t].unsqueeze(0)) #shape (num_models, 1,384)

                z_mean = z_pred.mean(dim=0) #shape (1,384)
                z = torch.cat((z[:,1:], z_mean.unsqueeze(0)), dim=1)
                
                predictions[:,t,:] =  z_pred.squeeze(1) #shape (num_models,384)
            else:
                z_pred = model(z, a_seq[t].unsqueeze(0)) #shape (1,384)
                predictions[t,:] = z_pred.squeeze(0)
                z = torch.cat((z[:,1:], z_pred.unsqueeze(0)), dim=1)

    return predictions        
