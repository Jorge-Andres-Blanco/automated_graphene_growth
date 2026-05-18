from pathlib import Path
import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
from data_processing.dinov2_encoder import DinoEncoder
from src.utils.plotting import compare_images_in_latent_space


# To be changed according to the executing machine
DATA_FILES = [
        (
         "/data/lmcat/inhouse/20260312/ihma818/id10-surf/20260301/RAW_DATA/CV_test_Gr_1_120326_camera/CV_test_Gr_1_120326_camera_0001",
         "CV_test_Gr_1_120326_camera_0001_with_experimental_data.h5",
            1
        ),
        (
         "/data/lmcat/inhouse/20260316/ihma818/id10-surf/20260301/RAW_DATA/CV_test_Gr_2_160326_camera/CV_test_Gr_2_160326_camera_0001",
         "CV_test_Gr_2_160326_camera_0001_with_experimental_data.h5",
            1
        ),
        (
         "/data/lmcat/inhouse/20260316/ihma818/id10-surf/20260301/RAW_DATA/CV_test_Gr_3_170326_camera/CV_test_Gr_3_170326_camera_0001",
         "CV_test_Gr_3_170326_camera_0001.h5",
            1
        ),
        ("/data/lmcat/inhouse/20260316/ihma818/id10-surf/20260301/RAW_DATA/Gr_4_080426_camera/Gr_4_080426_camera_0001/",
         "Gr_4_080426_camera_0001.h5",
            2
        ),
        (
         "/data/lmcat/inhouse/20260316/ihma818/id10-surf/20260301/RAW_DATA/Gr_5_090426_camera/Gr_5_090426_camera_0001/",
         "Gr_5_090426_camera_0001.h5",
            2
        ),
        (
         "/data/lmcat/inhouse/20260316/ihma818/id10-surf/20260301/RAW_DATA/Gr_6_100426_camera/Gr_6_100426_camera_0001/",
         "Gr_6_100426_camera_0001.h5",
            1
        ),
        (
         "/data/lmcat/inhouse/20260428/ihma832/id10-surf/20260401/RAW_DATA/Gr_1_280426_camera/Gr_1_280426_camera_0001/",
         "Gr_1_280426_camera_0001.h5",
            3
        )
    ]


def process_h5_with_dino(file_name: str, scan_number: str, encoder: DinoEncoder, save_file_name: str = None, sleep_time_basler: int = 2):
    """
    Streams images directly from an HDF5 file on disk to the GPU via the DINO encoder.
    """

    with h5py.File(file_name, "r") as f:
        
        dataset_path = f"{scan_number}.1/measurement/basler"

        
        if dataset_path not in f:
            raise KeyError(f"Path {dataset_path} not found in {file_name}")
            
        measurements = f[dataset_path]

        print(f"File: {file_name}")
        print(f"Found dataset -> Shape: {measurements.shape}, Type: {measurements.dtype}, Chunks: {measurements.chunks}")
        
        
        embeddings = encoder.encode_numpy_array(measurements, batch_size=16)

        if save_file_name is not None:
            np.save(file=save_file_name, arr=embeddings)
    
    return embeddings


def get_frame_from_h5(file_name:str, scan_number:str, frame_number:int):

    
    with h5py.File(file_name, "r") as f:
        
        dataset_path = f"{scan_number}.1/measurement/basler"

        
        if dataset_path not in f:
            raise KeyError(f"Path {dataset_path} not found in {file_name}")
            
        measurements = f[dataset_path]

        frame = measurements[frame_number]

    return frame


def get_frame_data(movie_num, frame_num):

    movie_path, file_name, scan_number = DATA_FILES[movie_num]

    full_file_path = os.path.join(movie_path, file_name)

    frame = get_frame_from_h5(full_file_path, scan_number, frame_num)

    return frame

def plot_2_frames(frame_0, frame_1):
    
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(frame_0)
    ax[1].imshow(frame_1)

    plt.show()

    return None

def encode_frames(frames, encoder: DinoEncoder):
    
    frames = np.concatenate(frames, axis=np.newaxis)
    embeddings = encoder.encode_numpy_array(frames, batch_size=16)
    return embeddings




    

def main():

    frame_0 = get_frame_data(0, 3050)
    frame_1 = get_frame_data(1, 8000)

    plot_2_frames(frame_0, frame_1)

    encoder = DinoEncoder()
    embeddings = encode_frames([frame_0, frame_1], encoder=encoder)
    z0, z1 = embeddings[0], embeddings[1]

    l2_distance = np.linalg.norm(z0-z1)
    cosine_similarity = np.dot(z0, z1) / (np.linalg.norm(z0) * np.linalg.norm(z1))
    compare_images_in_latent_space(frame_0, frame_1, z0, z1)


    print(f"L2 Distance: {l2_distance}")
    print(f"Cosine Similarity: {cosine_similarity}")

    return None


if __name__ == "__main__":

    main()