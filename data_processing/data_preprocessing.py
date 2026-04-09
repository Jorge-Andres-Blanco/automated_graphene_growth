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

        print(f"Found dataset -> Shape: {measurements.shape}, Type: {measurements.dtype}, Chunks: {measurements.chunks}")
        
        
        embeddings = encoder.encode_numpy_array(measurements, batch_size=16, save_file_name=save_file_name)

        if sleep_time_basler == 1:
        
            if file_name.endswith("Gr_4_080426_camera_0001.h5"):
                embeddings = embeddings[:2500:2]  # Downsample by taking every other element and cutting off the last part
                print(f"Downsampling images for {file_name} due to sleep_time_basler=1. Original shape: {embeddings.shape}, New shape: {embeddings.shape}")
            else:
                embeddings = embeddings[::2]
                np.save(file=save_file_name, arr=embeddings) # Overwrite with the downsampled version if sleep_time_basler is 1
                print(f"Downsampling images for {file_name} due to sleep_time_basler=1. Original shape: {embeddings.shape}, New shape: {embeddings.shape}")
    return embeddings


def extract_from_h5_to_npy(file_path: str, scan_number: str, measurement = 'CH4', save_file_name: str = None, sleep_time_basler: int = 2):

    with h5py.File(file_path, "r") as f:
        
        dataset_path = f"{scan_number}.1/measurement/{measurement}"

        data_np = f[dataset_path]

        if save_file_name is not None and sleep_time_basler == 2:
            np.save(file=save_file_name, arr=data_np)
        
        elif save_file_name is not None and sleep_time_basler == 1:
            
            if file_path.endswith("Gr_4_080426_camera_0001.h5"):
                data_np = data_np[:2500:2]  # Downsample by taking every other element and cutting off the last part
                print(f"Downsampling measurement data for {file_path} due to sleep_time_basler=1. Original shape: {data_np.shape}, New shape: {data_np.shape}")
            else:
                data_np = data_np[::2]  # Downsample by taking every other element
                print(f"Downsampling measurement data for {file_path} due to sleep_time_basler=1. Original shape: {data_np.shape}, New shape: {data_np.shape}")
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
        )
    ]

    saving_folder = "/data/lmcat/Computer_vision/data_arrays"


    
    measurement = "CH4"

    file_num = 0

    sleep_time_basler = 2

    for folder_path, file_name, scan in DATA_FILES:
    
        file_path = os.path.join(folder_path, file_name)

        movie_cls_path = os.path.join(saving_folder, f"movie_{file_num}_scan{scan}_cls.npy")
        save_measurement_path = os.path.join(saving_folder, f"measurement_{file_num}_{measurement}_scan{scan}.npy")

        if file_name == "Gr_4_080426_camera_0001.h5":
            sleep_time_basler = 1

        measurement_data = extract_from_h5_to_npy(file_path=file_path, scan_number=scan, measurement=measurement, save_file_name=save_measurement_path, sleep_time_basler=sleep_time_basler)

        embeddings = process_h5_with_dino(file_path, scan, encoder, save_file_name=movie_cls_path, sleep_time_basler=sleep_time_basler)
        
        print("Final Shape:\n", "embeddings: ", embeddings.shape, "measurement_data: ", measurement_data.shape, "sleep_time_basler: ", sleep_time_basler)
        

        sleep_time_basler = 2 # Reset to default for next file
        file_num += 1
