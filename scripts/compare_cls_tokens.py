from pathlib import Path
import h5py
import os
import matplotlib.pyplot as plt
from src.data_handling import HDF5Processor
import numpy as np
from src.models import DinoEncoder
from src.utils.plotting import compare_images_in_latent_space, plot_2_frames

def main():

    processor = HDF5Processor(encoder = DinoEncoder())

    frame_0 = processor.get_frame_data(0, 3050)
    frame_1 = processor.get_frame_data(1, 8000)

    plot_2_frames(frame_0, frame_1)

    embeddings = processor.encode_frames([frame_0, frame_1])
    z0, z1 = embeddings[0], embeddings[1]

    l2_distance = np.linalg.norm(z0-z1)
    cosine_similarity = np.dot(z0, z1) / (np.linalg.norm(z0) * np.linalg.norm(z1))
    compare_images_in_latent_space(frame_0, frame_1, z0, z1)


    print(f"L2 Distance: {l2_distance}")
    print(f"Cosine Similarity: {cosine_similarity}")

    return None


if __name__ == "__main__":

    main()