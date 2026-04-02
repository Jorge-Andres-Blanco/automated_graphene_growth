import numpy as np
from pathlib import Path

folder_path = Path(r"\\dfs\data\lmcat\Computer_vision\data_arrays")

data_file_list = list(folder_path.glob("*.npy"))

save_folder_training = "\\\\dfs\data\lmcat\Computer_vision\\training_data\\"
save_folder_validation = "\\\\dfs\data\lmcat\Computer_vision\\validation_data\\"



evaluation_data_dict = {
    "_0_": [(1250,1490),(1920,2070)],
    "_1_": [(2650,2800),(3390,3530)],
    "_2_": [(1050,1200),(1800,2000), (4650,4800)]
}

i_train_movie = 0
i_eval_movie = 0
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

                if "movie" in str(file_name):

                    np.save(save_folder_training+f"train_movie_{i_train_movie}.npy",train_data)
                    print(f"Saved train movie {i_train_movie} from {file_name}, start: {start_train}, stop: {start_eval}")
                    np.save(save_folder_validation+f"eval_movie_{i_eval_movie}.npy",eval_data)
                    print(f"Saved evaluation movie {i_eval_movie} from {file_name}, start: {start_eval}, stop: {stop_eval}")

                    i_train_movie += 1
                    i_eval_movie += 1
                
                else:
                    np.save(save_folder_training+f"train_CH4_{i_train_CH4}.npy",train_data)
                    print(f"Saved train CH4 {i_train_CH4} from {file_name}, start: {start_train}, stop: {start_eval}")
                    np.save(save_folder_validation+f"eval_CH4_{i_eval_CH4}.npy",eval_data)
                    print(f"Saved evaluation CH4 {i_eval_CH4} from {file_name}, start: {start_eval}, stop: {stop_eval}")

                    i_train_CH4 += 1
                    i_eval_CH4 += 1
                
                start_train = stop_eval

                if idx == (len(intervals)-1): #This is the last interval for evaluation

                    if start_train < data.shape[0]: #There is still data for training

                        train_data = data[start_train:]

                        if "movie" in str(file_name):

                            np.save(save_folder_training+f"train_movie_{i_train_movie}.npy",train_data)
                            print(f"Saved train movie {i_train_movie} from {file_name}, start: {start_train}, stop: {start_eval}")

                            i_train_movie += 1

                        else:
                            np.save(save_folder_training+f"train_CH4_{i_train_CH4}.npy",train_data)
                            print(f"Saved train CH4 {i_train_CH4} from {file_name}, start: {start_train}, stop: {start_eval}")

                            i_train_CH4 += 1