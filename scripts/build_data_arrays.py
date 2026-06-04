import os
from pathlib import Path

# Import your new classes from the src package
from src.models.dinov2_encoder import DinoEncoder
from src.data_handling import HDF5Processor



def main():

    # Inicialize the processor
    processor = HDF5Processor(encoder=DinoEncoder())

    saving_folder = Path("/data/lmcat/Computer_vision/data_arrays")
    saving_folder.mkdir(parents=True, exist_ok=True) # Ensure the folder exists
    
    measurement = "CH4"
    
    for file_num, (folder_path, file_name, scan, crop_idx) in enumerate(processor.data_files):
        
        full_file_path = os.path.join(folder_path, file_name)
        
        # Define save paths
        sequence_cls_path = saving_folder / f"sequence_{file_num}_scan{scan}_cls.npy"
        save_seq_measurement_path = saving_folder / f"seq_measurement_{file_num}_{measurement}_scan{scan}.npy"

        # Determine slicing logic based on the specific file
        sleep_time = 1 if "Gr_4_080426_camera_0001.h5" in file_name else 2
        

        print(f"--- Processing File {file_num}: {file_name} ---")

        # 3. Extract Measurement (CH4)
        measurement_data = processor.extract_measurement(
            file_path=full_file_path, 
            scan_number=scan, 
            measurement=measurement, 
            save_path=save_seq_measurement_path, 
            sleep_time_basler=sleep_time,
            crop_index=crop_idx
        )

        # 4. Extract Images and encode with DINO
        embeddings = processor.process_images_with_dino(
            file_path=full_file_path, 
            scan_number=scan, 
            save_path=sequence_cls_path, 
            sleep_time_basler=sleep_time,
            crop_index=crop_idx
        )
        
        print(f"Finished {file_name} | Embeddings: {embeddings.shape} | {measurement}: {measurement_data.shape}\n")

if __name__ == "__main__":
    main()