import numpy as np
from pathlib import Path
import h5py
import yaml
from typing import List, Dict, Tuple
from src.utils.misc import load_yaml_config

PROJECT_ROOT = Path(__file__).resolve().parents[1] # Adjust this if your project structure changes, this assumes the script is in 'scripts/data_prep' and the config.yaml is in the root of the project.
config_path = PROJECT_ROOT / 'config.yaml'


config = load_yaml_config(config_path)

folder_path = PROJECT_ROOT / "data_arrays"

save_folder_training = PROJECT_ROOT / "training_data"
save_folder_validation = PROJECT_ROOT / "validation_data"

save_folder_training.mkdir(parents=True, exist_ok=True)
save_folder_validation.mkdir(parents=True, exist_ok=True)


data_file_list = list(folder_path.glob("*.npy"))

"""
evaluation_data_dict: This dictionary defines the intervals for evaluation (and implicitly training) data for each file.
The structure is as follows:
evaluation_data_dict = {
"_N_": [(X,Y)],
...
}
Where:
N is the index of the movie/file in the DATA_FILES list (see hdf5_processor.py)
(X,Y) is a tuple defining the start (X) and stop (Y) indices for the evaluation data slice.
If X=Y, it means there is no evaluation data for that file, and all data will be used for training.
It is important to add all files that are intended for training even if they do not contain evaluation data, to ensure they are processed and saved correctly.
"""

evaluation_data_dict = {
    "_0_": [(1800,2200)],
    "_1_": [(2600,2900)],
    "_2_": [(1700,2100)],
    "_3_": [(0,0)],
    "_4_": [(0,500)],
    "_5_": [(0,0)],
    "_6_": [(0,0)],
    "_7_": [(0,0)],
    "_8_": [(0,0)],
    "_10_": [(0,0)],
    "_11_": [(0,0)]
}

i_train_sequence = 0
i_eval_sequence = 0
i_train_CH4 = 0
i_eval_CH4 = 0


for num, intervals in evaluation_data_dict.items():
    
    for file_name in data_file_list:

        if num in str(file_name):

            data = np.load(folder_path / file_name)
            start_train = 0
            
            for idx, (start_eval, stop_eval) in enumerate(intervals):
                
                train_data = data[start_train:start_eval]
                eval_data = data[start_eval:stop_eval]

                # Only save if the training slice actually has elements
                if train_data.shape[0] > 0:
                    if "sequence" in str(file_name):
                        np.save(save_folder_training+f"train_sequence_{i_train_sequence}.npy", train_data)
                        print(f"Saved train sequence {i_train_sequence} from {file_name}, start: {start_train}, stop: {start_eval}")
                        i_train_sequence += 1
                    else:
                        np.save(save_folder_training+f"train_CH4_{i_train_CH4}.npy", train_data)
                        print(f"Saved train CH4 {i_train_CH4} from {file_name}, start: {start_train}, stop: {start_eval}")
                        i_train_CH4 += 1

                # Only save if the evaluation slice actually has elements
                if eval_data.shape[0] > 0:
                    if "sequence" in str(file_name):
                        np.save(save_folder_validation+f"eval_sequence_{i_eval_sequence}.npy", eval_data)
                        print(f"Saved evaluation sequence {i_eval_sequence} from {file_name}, start: {start_eval}, stop: {stop_eval}")
                        i_eval_sequence += 1
                    else:
                        np.save(save_folder_validation+f"eval_CH4_{i_eval_CH4}.npy", eval_data)
                        print(f"Saved evaluation CH4 {i_eval_CH4} from {file_name}, start: {start_eval}, stop: {stop_eval}")
                        i_eval_CH4 += 1
                
                start_train = stop_eval

                if idx == (len(intervals)-1): #This is the last interval for evaluation

                    if start_train < data.shape[0]: #There is still data for training

                        train_data = data[start_train:]

                        if "sequence" in str(file_name):

                            np.save(save_folder_training+f"train_sequence_{i_train_sequence}.npy",train_data)
                            print(f"Saved train sequence {i_train_sequence} from {file_name}, start: {start_train}, stop: {data.shape[0]}")

                            i_train_sequence += 1

                        else:
                            np.save(save_folder_training+f"train_CH4_{i_train_CH4}.npy",train_data)
                            print(f"Saved train CH4 {i_train_CH4} from {file_name}, start: {start_train}, stop: {data.shape[0]}")

                            i_train_CH4 += 1