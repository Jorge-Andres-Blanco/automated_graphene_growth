import numpy as np
from pathlib import Path

def load_array_steps(file, step_size):
    
    data = np.load(file)

    return data[::step_size]



def load_transition_data(folder_path, step_size, hist_length):

    """
    ### Parameters:
        folder_path: path with the data files
        step_size: gap between one measurement and the next
        hist_length: number of measurements to give to the model

    ### Returns:
        z_n, a_n, y_n
            z_n np.array (N, hist_length,384) with the cls tokens
            a_n np.array (N, hist_length, 1) with the actions (CH4 values)
            z_n np.array (N, 384) with the next (target) class token 
    """

    cls_files = sorted(folder_path.glob("*movie*.npy"), key=(lambda p: int(p.stem.split('_')[-1])))
    CH4_files = sorted(folder_path.glob("*CH4*.npy"), key=(lambda p: int(p.stem.split('_')[-1])))

    #Safety check
    if len(cls_files) != len(CH4_files):
         raise ValueError(f"File count mismatch: {len(cls_files)} movie files vs {len(CH4_files)} CH4 files.")

    z_list, a_list, y_list = [], [], []

    for cls_file, CH4_file in zip(cls_files, CH4_files):

        #Safety check
        if cls_file.stem.split('_')[-1] != CH4_file.stem.split('_')[-1]:
             raise ValueError(f"Pairing mismatch! Trying to match {cls_file.name} with {CH4_file.name}")

        data_cls = load_array_steps(cls_file, step_size)
        data_CH4 = load_array_steps(CH4_file, step_size)

        N = data_cls.shape[0]

        print(cls_file, CH4_file, data_cls.shape, data_CH4.shape)

        #Safety check
        if N != data_CH4.shape[0]:
            raise ValueError(f"Length mismatch between {cls_file} and {CH4_file}. Probably they are not synchronized")


        #Make all possible combinations of tuples (z_0, a_0, z_1)
        for i in range(N-hist_length):
            
            z_0 = data_cls[i:i+hist_length]
            a_0 = data_CH4[i:i+hist_length]

            z_1 = data_cls[i+hist_length]

            z_list.append(z_0)
            a_list.append(a_0)
            y_list.append(z_1)

    return np.array(z_list), np.array(a_list), np.array(y_list)