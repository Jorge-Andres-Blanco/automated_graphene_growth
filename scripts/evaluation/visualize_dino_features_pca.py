from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models import DinoEncoder
from src.data_handling import HDF5Processor
from src.utils.plotting import adjust_exposure_gray_image

encoder = DinoEncoder()
data_processor = HDF5Processor(encoder=encoder)


# Image 1
movie_num = 7


frame_nums = [100, 150, 200, 250, 300, 350]  # Example frame numbers to visualize

frames = []

for frame_num in frame_nums:

    frame = data_processor.get_frame_data(movie_num, frame_num)
    frames.append(frame)

frames_np = np.array(frames)

# Path to save image
save_path = f"/data/lmcat/Computer_vision/automated_graphene_growth/plots/pca_features.png"

pca_images = encoder.visualize_pca(frames_np)

fig, axes = plt.subplots(6, 2, figsize=(6, 18))

for i, frame_num in enumerate(frame_nums):

    #axes[i, 0].imshow(frames_np[i], cmap='gray')
    frame_transformed_tensor = encoder.transform(torch.from_numpy(frames_np[i][np.newaxis, np.newaxis, :, :].astype(np.float32) / 4095.0).to(encoder.device))
    frame_transformed_np = frame_transformed_tensor.cpu().numpy()[0, 0, :, :]
    axes[i, 0].imshow(frame_transformed_np, cmap='gray')
    axes[i, 1].imshow(pca_images[i])

    axes[i,0].axis('off')
    axes[i,1].axis('off')

plt.tight_layout()
plt.savefig(save_path)