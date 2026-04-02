import numpy as np
import h5py
import hdf5plugin
import os

from dinov2_encoder import DinoEncoder


def process_h5_with_dino(file_name: str, scan_number: str, encoder: DinoEncoder, save_file_name: str = None):
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
        
    return embeddings


def extract_from_h5_to_npy(file_path: str, scan_number: str, measurement = 'CH4', save_file_name: str = None):

    with h5py.File(file_path, "r") as f:
        
        dataset_path = f"{scan_number}.1/measurement/{measurement}"

        data_np = f[dataset_path]

        if save_file_name is not None:
            np.save(file=save_file_name, arr=data_np)
        
        return data_np



if __name__ == "__main__":
    encoder = DinoEncoder()
    
    folder_path1 = "/data/lmcat/inhouse/20260312/ihma818/id10-surf/20260301/RAW_DATA/CV_test_Gr_1_120326_camera/CV_test_Gr_1_120326_camera_0001"
    folder_path2 = "/data/lmcat/inhouse/20260316/ihma818/id10-surf/20260301/RAW_DATA/CV_test_Gr_2_160326_camera/CV_test_Gr_2_160326_camera_0001"
    folder_path3 = "/data/lmcat/inhouse/20260316/ihma818/id10-surf/20260301/RAW_DATA/CV_test_Gr_3_170326_camera/CV_test_Gr_3_170326_camera_0001"


    file_name1 = "CV_test_Gr_1_120326_camera_0001_with_experimental_data.h5"
    file_name2 = "CV_test_Gr_2_160326_camera_0001_with_experimental_data.h5"
    file_name3 = "CV_test_Gr_3_170326_camera_0001.h5"

    scan1 = 1
    scan2 = 1
    scan3 = 1

    saving_folder = "/data/lmcat/Computer_vision/data_arrays"


    files = [(folder_path1, file_name1, scan1), (folder_path2, file_name2, scan2), (folder_path3, file_name3, scan3)]
    measurement = "CH4"

    file_num = 0

    for folder_path, file_name, scan in files:
    
        file_path = os.path.join(folder_path, file_name)

        movie_cls_path = os.path.join(saving_folder, f"movie_{file_num}_scan{scan}_cls.npy")
        save_measurement_path = os.path.join(saving_folder, f"measurement_{file_num}_{measurement}_scan{scan}.npy")

        measurement_data = extract_from_h5_to_npy(file_path=file_path, scan_number=scan, measurement=measurement, save_file_name=save_measurement_path)

        embeddings = process_h5_with_dino(file_path, scan, encoder, save_file_name=movie_cls_path)
        
        print("Final Shape:\n", "embeddings: ", embeddings.shape, "measurement_data: ", measurement_data.shape)
        file_num += 1