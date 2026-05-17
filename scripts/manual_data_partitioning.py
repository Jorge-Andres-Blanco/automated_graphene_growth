import numpy as np
from pathlib import Path

folder_path = Path(r"\\dfs\data\lmcat\Computer_vision\data_arrays")

data_file_list = list(folder_path.glob("*.npy"))

save_folder_training = "\\\\dfs\data\lmcat\Computer_vision\\training_data\\"
save_folder_validation = "\\\\dfs\data\lmcat\Computer_vision\\validation_data\\"



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