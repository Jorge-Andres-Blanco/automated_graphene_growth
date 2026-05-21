import numpy as np
import matplotlib.pyplot as plt


def plot_2_frames(frame_0, frame_1):
    
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(frame_0)
    ax[1].imshow(frame_1)

    plt.show()

    return None

def compare_images_in_latent_space(img1, img2, z1, z2, cmap='gray'):
    """
    Plots two images side-by-side and displays their L2 distance 
    and cosine similarity in the latent space.

    Args:
        img1 (np.ndarray): The first image array (e.g., from the Basler camera).
        img2 (np.ndarray): The second image array.
        z1 (np.ndarray): The DINOv2 latent embedding for the first image. Shape: (384,)
        z2 (np.ndarray): The DINOv2 latent embedding for the second image. Shape: (384,)
        cmap (str, optional): Colormap for matplotlib. Defaults to 'gray'.
    """
    
    # Ensure the embeddings are 1D arrays
    z1 = np.squeeze(z1)
    z2 = np.squeeze(z2)

    # Calculate L2 Distance
    l2_distance = np.linalg.norm(z1 - z2)

    # Calculate Cosine Similarity
    # Formula: (A dot B) / (||A|| * ||B||)
    dot_product = np.dot(z1, z2)
    norm_z1 = np.linalg.norm(z1)
    norm_z2 = np.linalg.norm(z2)
    
    # Avoid division by zero just in case
    if norm_z1 == 0 or norm_z2 == 0:
        cos_similarity = 0.0
    else:
        cos_similarity = dot_product / (norm_z1 * norm_z2)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Image 1
    axes[0].imshow(img1, cmap=cmap)
    axes[0].set_title("Image 1", fontsize=14)
    axes[0].axis('off') # Hide axes for cleaner image view

    # Plot Image 2
    axes[1].imshow(img2, cmap=cmap)
    axes[1].set_title("Image 2", fontsize=14)
    axes[1].axis('off')

    # Add the metrics as a main title above the images
    title_text = (
        f"Latent Space Comparison\n"
        f"L2 Distance: {l2_distance:.4f}  |  Cosine Similarity: {cos_similarity:.4f}"
    )
    plt.suptitle(title_text, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"comparison_img_and_tokens.png")