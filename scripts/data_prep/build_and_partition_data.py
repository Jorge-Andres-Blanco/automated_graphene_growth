import os
import yaml
import numpy as np
from pathlib import Path

from src.models.dinov2_encoder import DinoEncoder
from src.data_handling import HDF5Processor
from src.utils.misc import load_yaml_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Adjust this if project structure changes, this assumes the script is in 'scripts/data_prep' and the config.yaml is in the root of the project.


def main():
    config_path = PROJECT_ROOT / 'config.yaml'
    config = load_yaml_config(config_path)
    
    # Initialize the processor (DINO is utilized strictly for inference)
    processor = HDF5Processor(encoder=DinoEncoder())

    # Output directories
    intermediate_folder = PROJECT_ROOT / "data_arrays"
    save_folder_training = PROJECT_ROOT / "data_processing" / "training_data"
    save_folder_validation = PROJECT_ROOT / "data_processing" / "validation_data"
    
    intermediate_folder.mkdir(parents=True, exist_ok=True, mode = 0o777)
    save_folder_training.mkdir(parents=True, exist_ok=True, mode = 0o777)
    save_folder_validation.mkdir(parents=True, exist_ok=True, mode = 0o777)

    measurement = "CH4"
    data_files = config.get('data_files', [])

    for item in data_files:
        folder_path = item['path']
        file_name = item['file_name']
        scan = item['scan_number']
        crop_idx = item['crop_index']
        
        train_ranges = item.get('training_ranges', [])
        val_ranges = item.get('validation_ranges', [])

        full_file_path = os.path.join(folder_path, file_name)
        
        # Use the file's base name (without .h5) as the unique identifier
        base_name = Path(file_name).stem 
        
        file_scan_id = f"{base_name}_scan{scan}"
        
        sequence_cls_path = intermediate_folder / f"{file_scan_id}_embeddings.npy"
        save_seq_measurement_path = intermediate_folder / f"{file_scan_id}_{measurement}.npy"

        sleep_time = 1 if "Gr_4_080426_camera_0001.h5" in file_name else 2

        print(f"\n--- Processing & Partitioning: {file_scan_id} ---")

        if sequence_cls_path.exists() and save_seq_measurement_path.exists():
            print("  -> Intermediate arrays found. Loading directly from disk...")
            measurement_data = np.load(save_seq_measurement_path)
            embeddings = np.load(sequence_cls_path)
        else:
            print("  -> Intermediate arrays not found. Running extraction and DINO inference...")
            measurement_data = processor.extract_measurement(
                file_path=full_file_path, 
                scan_number=scan, 
                measurement=measurement, 
                save_path=save_seq_measurement_path, 
                sleep_time_basler=sleep_time,
                crop_index=crop_idx
            )
            embeddings = processor.process_images_with_dino(
                file_path=full_file_path, 
                scan_number=scan, 
                save_path=sequence_cls_path, 
                sleep_time_basler=sleep_time,
                crop_index=crop_idx
            )
        
        print(f"  -> Data Ready | Embeddings: {embeddings.shape} | {measurement}: {measurement_data.shape}")

        # Partition Training Data (incorporating file_scan_id)
        for chunk_idx, (start_idx, end_idx) in enumerate(train_ranges):
            if start_idx == end_idx:
                continue
                
            train_seq = embeddings[start_idx:end_idx]
            train_ch4 = measurement_data[start_idx:end_idx]
            
            if train_seq.shape[0] > 0:
                np.save(save_folder_training / f"train_seq_{file_scan_id}_chunk{chunk_idx}.npy", train_seq)
                print(f"  -> Saved train sequence: train_seq_{file_scan_id}_chunk{chunk_idx}.npy")
                
            if train_ch4.shape[0] > 0:
                np.save(save_folder_training / f"train_{measurement}_{file_scan_id}_chunk{chunk_idx}.npy", train_ch4)
                print(f"  -> Saved train {measurement}: train_{measurement}_{file_scan_id}_chunk{chunk_idx}.npy")

        # Partition Validation Data (incorporating file_scan_id)
        for chunk_idx, (start_idx, end_idx) in enumerate(val_ranges):
            if start_idx == end_idx:
                continue
                
            eval_seq = embeddings[start_idx:end_idx]
            eval_ch4 = measurement_data[start_idx:end_idx]
            
            if eval_seq.shape[0] > 0:
                np.save(save_folder_validation / f"eval_seq_{file_scan_id}_chunk{chunk_idx}.npy", eval_seq)
                print(f"  -> Saved eval sequence: eval_seq_{file_scan_id}_chunk{chunk_idx}.npy")
                
            if eval_ch4.shape[0] > 0:
                np.save(save_folder_validation / f"eval_{measurement}_{file_scan_id}_chunk{chunk_idx}.npy", eval_ch4)
                print(f"  -> Saved eval {measurement}: eval_{measurement}_{file_scan_id}_chunk{chunk_idx}.npy")

    print("\nExtraction and Partitioning Pipeline Complete.")

if __name__ == "__main__":
    main()