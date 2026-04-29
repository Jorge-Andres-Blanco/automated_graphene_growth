import numpy as np
import h5py
import hdf5plugin
import os

from dinov2_encoder import DinoEncoder

"""
Structure of the data files:
    [
    (
    "path_to_data_file",
    "file_name",
    scan_number
    ),
    (
    "another_path",
    "another_file_name",
    scan_number
    )
    ]
"""

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
        ),
        (
         "/data/lmcat/inhouse/20260428/ihma832/id10-surf/20260401/RAW_DATA/Gr_1_280426_camera/Gr_1_280426_camera_0001/",
         "Gr_1_280426_camera_0001.h5",
            4
        ),
        (
         "/data/lmcat/inhouse/20260428/ihma832/id10-surf/20260401/RAW_DATA/Gr_1_280426_camera/Gr_1_280426_camera_0001/",
         "Gr_1_280426_camera_0001.h5",
            5
        )
    ]

def slice_data_with_conditions(data, file_name, scan_number, save_file_name=None, sleep_time_basler=2):
    """
    Applies specific temporal slicing or downsampling to the data array based on the 
    filename and recording parameters to remove irrelevant parts (e.g., reactor cooling).

    Parameters
    ----------
    data : np.ndarray or h5py.Dataset
        The input data sequence. Shape: (Time, ...).
    file_name : str
        The name of the file being processed. Used to match specific hardcoded conditions.
    scan_number : int
        The scan number, used to differentiate slices within the same file.
    save_file_name : str, optional
        If provided, the sliced NumPy array will be saved to this path. Defaults to None.
    sleep_time_basler : int, optional
        The sleep time parameter used during recording. Determines the downsampling logic. Defaults to 2.

    Returns
    -------
    np.ndarray
        The sliced/downsampled data. Shape: (New_Time, ...).
        
    Hardcoded Elements
    ------------------
    - Slice bounds: Specific filenames are mapped to hardcoded crop indices (e.g., 3047, 6600, 7015).
    - Downsampling logic: If `sleep_time_basler == 1`, data is downsampled by a factor of 2 (`[::2]`).
    """
    if sleep_time_basler == 2:

        if file_name.endswith("CV_test_Gr_1_120326_camera_0001_with_experimental_data.h5"): #Remove last 15 minutes of reactor cooling
            data = data[:3047]

        elif file_name.endswith("CV_test_Gr_2_160326_camera_0001_with_experimental_data.h5"): # Remove from 14:32 due to stationary image
            data = data[:6600]
        
        elif file_name.endswith("CV_test_Gr_3_170326_camera_0001.h5"): # Remove last 35 minutes of reactor cooling
            data = data[:7015]
        elif file_name.endswith("Gr_1_280426_camera_0001.h5"): # Remove last 35 minutes of reactor cooling
            if scan_number == 4:
                data = data[:1900]
            elif scan_number == 5:
                data = data[:820]
    
    elif sleep_time_basler == 1:
        
        if file_name.endswith("Gr_4_080426_camera_0001.h5"):
            data = data[:2500:2]  # Downsample by taking every other element and cutting off the last part
        else:
            data = data[::2]  # Downsample by taking every other element
        
    if save_file_name is not None:
        np.save(file=save_file_name, arr=data)

    else:
        print("No save file name provided, or sleep_time_basler is not 2 or 1.\n"
                "Skipping saving the measurement data.")    

    return data




def process_h5_with_dino(file_name: str, scan_number: str, encoder: DinoEncoder, save_file_name: str = None, sleep_time_basler: int = 2):
    """
    Reads image frames from an HDF5 dataset and passes them through a DINOv2 encoder 
    to extract latent embeddings, then applies condition-based slicing.

    Parameters
    ----------
    file_name : str
        Path to the HDF5 file.
    scan_number : str or int
        The scan number used to locate the dataset inside the HDF5 file.
    encoder : DinoEncoder
        The initialized DINOv2 encoder instance.
    save_file_name : str, optional
        Path to save the extracted embeddings as a .npy file. Defaults to None.
    sleep_time_basler : int, optional
        The sleep time parameter used during recording. Defaults to 2.

    Returns
    -------
    embeddings : np.ndarray
        The extracted and sliced latent embeddings. Shape: (Time, 384).

    Hardcoded Elements
    ------------------
    - Dataset path: Hardcoded to `"{scan_number}.1/measurement/basler"`.
    - Batch size: Hardcoded to `16` for the `encode_numpy_array` method.
    """

    with h5py.File(file_name, "r") as f:
        
        dataset_path = f"{scan_number}.1/measurement/basler"

        
        if dataset_path not in f:
            raise KeyError(f"Path {dataset_path} not found in {file_name}")
            
        measurements = f[dataset_path]

        print(f"File: {file_name}")
        print(f"Found dataset -> Shape: {measurements.shape}, Type: {measurements.dtype}, Chunks: {measurements.chunks}")
        
        
        embeddings = encoder.encode_numpy_array(measurements, batch_size=16)

        # File-specific downsampling
        embeddings = slice_data_with_conditions(embeddings, file_name, scan_number, save_file_name, sleep_time_basler)
    
    return embeddings


def extract_from_h5_to_npy(file_name: str, scan_number: str, measurement = 'CH4', save_file_name: str = None, sleep_time_basler: int = 2):
    """
    Extracts a 1D measurement sequence (such as CH4 flow) from an HDF5 file and 
    applies necessary temporal slicing/downsampling.

    Parameters
    ----------
    file_name : str
        Path to the HDF5 file.
    scan_number : str or int
        The scan number used to locate the dataset inside the HDF5 file.
    measurement : str, optional
        The name of the measurement to extract. Defaults to 'CH4'.
    save_file_name : str, optional
        Path to save the extracted array as a .npy file. Defaults to None.
    sleep_time_basler : int, optional
        The sleep time parameter used during recording. Defaults to 2.

    Returns
    -------
    data_np : np.ndarray
        The extracted and sliced 1D measurement array. Shape: (Time,).

    Hardcoded Elements
    ------------------
    - Dataset path: Hardcoded to `"{scan_number}.1/measurement/{measurement}"`.
    """

    with h5py.File(file_name, "r") as f:
        
        dataset_path = f"{scan_number}.1/measurement/{measurement}"

        data_np = f[dataset_path]

        # File-specific conditions
        data_np = slice_data_with_conditions(data_np, file_name, scan_number, save_file_name, sleep_time_basler=sleep_time_basler)
        
        return data_np



if __name__ == "__main__":
    encoder = DinoEncoder()

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