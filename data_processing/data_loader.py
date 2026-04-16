import numpy as np
import torch
from pathlib import Path

def get_npy_file_shape(file_path):
    """
    Reads the shape of a .npy file instantly without loading it into RAM.
    """
    # 'r' opens it in read-only memory-mapped mode
    mmap_array = np.load(file_path, mmap_mode='r')
    
    return mmap_array.shape


def load_transition_data(folder_path, step_size, hist_length, return_indices = False):

    """
    ### Parameters:
        folder_path: path with the data files
        step_size: gap between one measurement and the next
        hist_length: number of measurements to give to the model
        return_indices: returns the indices (start, stop) for each sequence.

    ### Returns:
        z_n, a_n, y_n
            z_n np.array (N, hist_length,384) with the cls tokens
            a_n np.array (N, hist_length, 1) with the actions (CH4 values)
            z_n np.array (N, 384) with the next (target) class token 
    """

    cls_files = sorted(folder_path.glob("*sequence*.npy"), key=(lambda p: int(p.stem.split('_')[-1])))
    CH4_files = sorted(folder_path.glob("*CH4*.npy"), key=(lambda p: int(p.stem.split('_')[-1])))

    #Safety check
    if len(cls_files) != len(CH4_files):
         raise ValueError(f"File count mismatch: {len(cls_files)} sequence files vs {len(CH4_files)} CH4 files.")

    z_list, a_list, y_list = [], [], []
    indices = []
    index = 0
    for cls_file, CH4_file in zip(cls_files, CH4_files):

        #Safety check
        if cls_file.stem.split('_')[-1] != CH4_file.stem.split('_')[-1]:
             raise ValueError(f"Pairing mismatch: {cls_file.name} with {CH4_file.name}")

        data_cls = np.load(cls_file)
        data_CH4 = np.load(CH4_file)

        N = data_cls.shape[0]

        #Safety check
        if N != data_CH4.shape[0]:
            raise ValueError(f"Length mismatch between {cls_file} and {CH4_file}. Probably they are not synchronized")


        #Make all possible combinations of tuples (z_0, a_0, z_1)
        max_idx = N - (hist_length * step_size) 
        
        for i in range(max_idx):
            
            # Slice with a step! [start : stop : step]
            z_0 = data_cls[i : i + (hist_length * step_size) : step_size]
            a_0 = data_CH4[i : i + (hist_length * step_size) : step_size]

            # The target is the next step_size jump after the history ends
            z_1 = data_cls[i + (hist_length * step_size)]

            z_list.append(z_0)
            a_list.append(a_0)
            y_list.append(z_1)

        
        indices.append((index, index+(N-hist_length)))
        index = indices[-1][1]

    if return_indices:
        return np.array(z_list), np.array(a_list), np.array(y_list), indices

    return np.array(z_list), np.array(a_list), np.array(y_list)




def get_scenes_from_files(cls_files, hist_length, step_size):
    """
        Missing documentation
    """

    len_files_array = np.array([get_npy_file_shape(f)[0] for f in cls_files])

    min_len_file = np.min(len_files_array)

    # Length of the scenes
    scenes_len = min_len_file

    indices_scenes = []
    file = 0
    context_needed = hist_length * step_size


    for cls_seq in cls_files:

        cls_seq_len = get_npy_file_shape(cls_seq)[0]

        scenes_in_seq = round(cls_seq_len/scenes_len+0.499) # This can be improved

        for i in range(scenes_in_seq):

            start, stop = i, i+scenes_len
            indices = (file, start, stop)
            
            if stop > cls_seq_len:
                this_scene_len = (cls_seq_len-start)
                if this_scene_len < context_needed:
                    break

            indices_scenes.append(indices)
        
        file += 1
    
    return indices_scenes
        


            
def load_transition_data_from_scenes(scene_path, hist_length, step_size):

    cls_files = sorted(scene_path.glob("*sequence*.npy"), key=(lambda p: int(p.stem.split('_')[-1])))
    CH4_files = sorted(scene_path.glob("*CH4*.npy"), key=(lambda p: int(p.stem.split('_')[-1])))

    #Safety check
    if len(cls_files) != len(CH4_files):
         raise ValueError(f"File count mismatch: {len(cls_files)} sequence files vs {len(CH4_files)} CH4 files.")

    z_list, a_list, y_list = [], [], []
    
    chosen_files = torch.randint(0, len(cls_files), len(cls_files))

    for f in chosen_files:

        cls_file = cls_files[f]
        CH4_file = CH4_files[f]

        #Safety check
        if cls_file.stem.split('_')[-1] != CH4_file.stem.split('_')[-1]:
             raise ValueError(f"Pairing mismatch: {cls_file.name} with {CH4_file.name}")

        data_cls = np.load(cls_file)
        data_CH4 = np.load(CH4_file)

        N = data_cls.shape[0]

        #Safety check
        if N != data_CH4.shape[0]:
            raise ValueError(f"Length mismatch between {cls_file} and {CH4_file}. Probably they are not synchronized")
        
        max_idx = N - (hist_length * step_size) 
        
        for i in range(max_idx):
            
            # Slice with a step! [start : stop : step]
            z_0 = data_cls[i : i + (hist_length * step_size) : step_size]
            a_0 = data_CH4[i : i + (hist_length * step_size) : step_size]

            # The target is the next step_size jump after the history ends
            z_1 = data_cls[i + (hist_length * step_size)]

            z_list.append(z_0)
            a_list.append(a_0)
            y_list.append(z_1)

    return np.array(z_list), np.array(a_list), np.array(y_list)
