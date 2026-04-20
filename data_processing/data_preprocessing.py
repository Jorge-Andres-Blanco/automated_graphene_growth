import numpy as np
import h5py
import hdf5plugin
import os

from dinov2_encoder import DinoEncoder


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

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Hardcoded downsampling for specific files
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        if sleep_time_basler == 1:
        
            if file_name.endswith("Gr_4_080426_camera_0001.h5"):
                embeddings = embeddings[:2500:2]  # Downsample by taking every other element and cutting off the last part
            else:
                embeddings = embeddings[::2]

        elif sleep_time_basler == 2:
            if file_name.endswith("CV_test_Gr_1_120326_camera_0001_with_experimental_data.h5"): #Remove last 15 minutes of reactor cooling
                embeddings = embeddings[:3047]

            elif file_name.endswith("CV_test_Gr_2_160326_camera_0001_with_experimental_data.h5"): # Remove from 14:32 due to stationary image
                embeddings = embeddings[:6600]
            
            elif file_name.endswith("CV_test_Gr_3_170326_camera_0001.h5"): # Remove last 35 minutes of reactor cooling
                embeddings = embeddings[:7015]


        if save_file_name is not None:
            np.save(file=save_file_name, arr=embeddings)
    
    return embeddings


def extract_from_h5_to_npy(file_name: str, scan_number: str, measurement = 'CH4', save_file_name: str = None, sleep_time_basler: int = 2):

    with h5py.File(file_name, "r") as f:
        
        dataset_path = f"{scan_number}.1/measurement/{measurement}"

        data_np = f[dataset_path]

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Hardcoded downsampling for specific files
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------

        if save_file_name is not None and sleep_time_basler == 2:

            if file_name.endswith("CV_test_Gr_1_120326_camera_0001_with_experimental_data.h5"): #Remove last 15 minutes of reactor cooling
                data_np = data_np[:3047]

            elif file_name.endswith("CV_test_Gr_2_160326_camera_0001_with_experimental_data.h5"): # Remove from 14:32 due to stationary image
                data_np = data_np[:6600]
            
            elif file_name.endswith("CV_test_Gr_3_170326_camera_0001.h5"): # Remove last 35 minutes of reactor cooling
                data_np = data_np[:7015]
            
            np.save(file=save_file_name, arr=data_np)
        
        elif save_file_name is not None and sleep_time_basler == 1:
            
            if file_name.endswith("Gr_4_080426_camera_0001.h5"):
                data_np = data_np[:2500:2]  # Downsample by taking every other element and cutting off the last part
            else:
                data_np = data_np[::2]  # Downsample by taking every other element
            
            np.save(file=save_file_name, arr=data_np)

        else:
            print("No save file name provided, or sleep_time_basler is not 2 or 1.\n"
                  "Skipping saving the measurement data.")
        
        return data_np



if __name__ == "__main__":
    encoder = DinoEncoder()

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
        )
    ]

    saving_folder = "/data/lmcat/Computer_vision/data_arrays"
    measurement = "CH4"
    file_num = 0

    for folder_path, file_name, scan in DATA_FILES:
        
        full_file_path = os.path.join(folder_path, file_name)

        sequence_cls_path = os.path.join(saving_folder, f"sequence_{file_num}_scan{scan}_cls.npy")
        save_seq_measurement_path = os.path.join(saving_folder, f"seq_measurement_{file_num}_{measurement}_scan{scan}.npy")

        if full_file_path.endswith("Gr_4_080426_camera_0001.h5"):
            current_sleep_time = 1
        else:
            current_sleep_time = 2

        measurement_data = extract_from_h5_to_npy(
            file_name=full_file_path, 
            scan_number=scan, 
            measurement=measurement, 
            save_file_name=save_seq_measurement_path, 
            sleep_time_basler=current_sleep_time
        )
        measurement_data = extract_from_h5_to_npy(file_path=full_file_path, scan_number=scan, measurement=measurement, save_file_name=save_seq_measurement_path, sleep_time_basler=current_sleep_time)

        embeddings = process_h5_with_dino(full_file_path, scan, encoder, save_file_name=sequence_cls_path, sleep_time_basler=current_sleep_time)
        
        print(f"File {file_num} ({file_name}) Final Shape:")
        print(f"  embeddings: {embeddings.shape}")
        print(f"  measurement_data: {measurement_data.shape}")
        print(f"  sleep_time_basler: {current_sleep_time}\n")
        
        file_num += 1