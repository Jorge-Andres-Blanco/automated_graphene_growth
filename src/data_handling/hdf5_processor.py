import numpy as np
import h5py
import hdf5plugin
import os
from src.models.dinov2_encoder import DinoEncoder


"""
Structure of the data files:
    [
    (
    "path_to_data_file",
    "file_name",
    scan_number,
    crop_index
    ),
    ...
    ]
"""

DATA_FILES = [
        (
         "/data/lmcat/inhouse/20260312/ihma818/id10-surf/20260301/RAW_DATA/CV_test_Gr_1_120326_camera/CV_test_Gr_1_120326_camera_0001",
         "CV_test_Gr_1_120326_camera_0001_with_experimental_data.h5",
            1,
            3047
        ),
        (
         "/data/lmcat/inhouse/20260316/ihma818/id10-surf/20260301/RAW_DATA/CV_test_Gr_2_160326_camera/CV_test_Gr_2_160326_camera_0001",
         "CV_test_Gr_2_160326_camera_0001_with_experimental_data.h5",
            1,
            6600
        ),
        (
         "/data/lmcat/inhouse/20260316/ihma818/id10-surf/20260301/RAW_DATA/CV_test_Gr_3_170326_camera/CV_test_Gr_3_170326_camera_0001",
         "CV_test_Gr_3_170326_camera_0001.h5",
            1,
            7015
        ),
        ("/data/lmcat/inhouse/20260316/ihma818/id10-surf/20260301/RAW_DATA/Gr_4_080426_camera/Gr_4_080426_camera_0001/",
         "Gr_4_080426_camera_0001.h5",
            2,
            2500
        ),
        (
         "/data/lmcat/inhouse/20260316/ihma818/id10-surf/20260301/RAW_DATA/Gr_5_090426_camera/Gr_5_090426_camera_0001/",
         "Gr_5_090426_camera_0001.h5",
            2,
            None
        ),
        (
         "/data/lmcat/inhouse/20260316/ihma818/id10-surf/20260301/RAW_DATA/Gr_6_100426_camera/Gr_6_100426_camera_0001/",
         "Gr_6_100426_camera_0001.h5",
            1,
            None
        ),
        (
         "/data/lmcat/inhouse/20260428/ihma832/id10-surf/20260401/RAW_DATA/Gr_1_280426_camera/Gr_1_280426_camera_0001/",
         "Gr_1_280426_camera_0001.h5",
            3,
            None
        ),
        (
         "/data/lmcat/inhouse/20260428/ihma832/id10-surf/20260401/RAW_DATA/Gr_1_280426_camera/Gr_1_280426_camera_0001/",
         "Gr_1_280426_camera_0001.h5",
            4,
            1900
        ),
        (
         "/data/lmcat/inhouse/20260428/ihma832/id10-surf/20260401/RAW_DATA/Gr_1_280426_camera/Gr_1_280426_camera_0001/",
         "Gr_1_280426_camera_0001.h5",
            5,
            820
        )
    ]


class HDF5Processor:
    """
    Handles extraction and temporal slicing of HDF5 measurement data and images.
    """
    

    def __init__(self, encoder:DinoEncoder=None, data_files=DATA_FILES):
        """
        Initializes the data loader.
        
        Parameters
        ----------
        data_files : list of tuples, optional
            A list containing (directory_path, file_name, scan_number, crop_index).
        encoder : DinoEncoder, optional
            An initialized DINO encoder for processing images.
        """
        self.encoder = encoder
        
        self.data_files = data_files

    
    def slice_data(self, data, sleep_time_basler=2, crop_index=None, step_downsample=None):
        """
        Applies temporal slicing or downsampling to the data array.
        Instead of hardcoding filenames, we pass the exact crop/downsample rules.
        """
        sliced_data = data
        
        # 1. Apply cropping if specified (e.g., to remove cooling phases)
        if crop_index is not None:
            sliced_data = sliced_data[:crop_index]
            
        # 2. Apply downsampling based on sleep time
        if sleep_time_basler == 1:
            if step_downsample:
                sliced_data = sliced_data[:step_downsample:2] 
            else:
                sliced_data = sliced_data[::2]
                
        return sliced_data

    
    def extract_measurement(self, file_path, scan_number, measurement='CH4', 
                            save_path=None, **slice_kwargs):
        """
        Extracts a 1D measurement sequence (like CH4) from an HDF5 file.
        """
        with h5py.File(file_path, "r") as f:
            dataset_path = f"{scan_number}.1/measurement/{measurement}"
            
            if dataset_path not in f:
                raise KeyError(f"Dataset path {dataset_path} not found in {file_path}")
                
            data_np = np.array(f[dataset_path])

            # Apply slicing
            data_np = self.slice_data(data_np, **slice_kwargs)
            
            if save_path:
                np.save(file=save_path, arr=data_np)
                print(f"Saved measurement to {save_path}")

            return data_np

    
    def process_images_with_dino(self, file_path, scan_number, batch_size=16, 
                                 save_path=None, **slice_kwargs):
        """
        Reads image frames and passes them through the DINO encoder.
        """
        if not self.encoder:
            raise ValueError("An encoder instance must be provided during initialization to process images.")

        with h5py.File(file_path, "r") as f:
            dataset_path = f"{scan_number}.1/measurement/basler"
            
            if dataset_path not in f:
                raise KeyError(f"Path {dataset_path} not found in {file_path}")
                
            measurements = f[dataset_path]
            print(f"Processing {file_path} -> Shape: {measurements.shape}")
            
            # Extract embeddings
            embeddings = self.encoder.encode_numpy_array(measurements, batch_size=batch_size)

            # Apply slicing
            embeddings = self.slice_data(embeddings, **slice_kwargs)
            
            if save_path:
                np.save(file=save_path, arr=embeddings)
                print(f"Saved embeddings to {save_path}")
        
        return embeddings
    
    def encode_frames(self, frames):
        frames = np.stack(frames, axis=0)
        embeddings = self.encoder.encode_numpy_array(frames, batch_size=16)
        return embeddings


    @staticmethod
    def get_frame_from_h5(file_name:str, scan_number:str, frame_number:int, measurement:str):

        with h5py.File(file_name, "r") as f:
            
            dataset_path = f"{scan_number}.1/measurement/{measurement}"

            
            if dataset_path not in f:
                raise KeyError(f"Path {dataset_path} not found in {file_name}")
                
            measurements = f[dataset_path]

            frame = measurements[frame_number]

        return frame


    def get_frame_data(self, movie_num, frame_num, measurement = "basler"):

        movie_path, file_name, scan_number, _ = self.data_files[movie_num]

        full_file_path = os.path.join(movie_path, file_name)

        frame = self.get_frame_from_h5(full_file_path, scan_number, frame_num, measurement=measurement)

        return frame