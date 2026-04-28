import numpy as np
import torch
from pathlib import Path

def get_npy_file_shape(file_path):
    """
    Reads the shape of a .npy file instantly without loading it into RAM.

    Parameters
    ----------
    file_path : str or pathlib.Path
        The absolute or relative path to the target .npy file.

    Returns
    -------
    shape : tuple
        A tuple representing the dimensions of the NumPy array stored in the file.

    Hardcoded Elements
    ------------------
    - mmap_mode: Hardcoded to 'r' (read-only memory-mapped mode) to prevent 
      modifications and avoid loading the entire array into memory.
    """
    # 'r' opens it in read-only memory-mapped mode
    mmap_array = np.load(file_path, mmap_mode='r')
    
    return mmap_array.shape


def load_transition_data(folder_path, step_size, hist_length, return_indices=False):

    """
    Loads transition data from sequence and action .npy files located in a directory.

    Parameters
    ----------
    folder_path : str or pathlib.Path
        Path to the directory containing the data files.
    step_size : int
        The gap (stride) between one measurement and the next.
    hist_length : int
        The number of historical measurements to provide as context to the model.
    return_indices : bool, optional
        If True, returns the logical indices (start, stop) for each sequence chunk. 
        Defaults to False.

    Returns
    -------
    z_n : np.ndarray
        The context cls tokens. Shape: (N, hist_length, 384).
    a_n : np.ndarray
        The context actions (CH4 values). Shape: (N, hist_length, 1).
    y_n : np.ndarray
        The next (target) class token. Shape: (N, 384).
    indices : list of tuples, optional
        A list of tuples formatted as `(start_index, stop_index)` mapping each sequence.
        Returned only if `return_indices=True`.

    Hardcoded Elements
    ------------------
    - Latent dimension: Assumed to be 384 based on the DINOv2 embeddings shape.
    - Action dimension: Assumed to be 1 (CH4 flow).
    - File matching: Hardcoded to glob for "*sequence*.npy" and "*CH4*.npy" files.
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




def get_scenes_indices_from_files(cls_files, hist_length, step_size):
    """
    Generates dataset sequence indices to split variable-length files into uniform scenes.
    
    Files are divided into chunks of length `scenes_len` (determined by the shortest 
    file in the dataset). Any remainder chunk at the end of a file is included only 
    if it satisfies the `context_needed` minimum length.

    Parameters
    ----------
    cls_files : list of str or pathlib.Path
        Ordered iterable containing the file paths of the cls tokens.
    hist_length : int
        The number of historical measurements required for the model input.
    step_size : int
        The gap between consecutive measurements in the sequence.

    Returns
    -------
    list of tuples
        A list of indices formatted as `(file_num, start_index, stop_index)` mapping
        valid sequence chunks for the dataloader.
        
    Hardcoded Elements
    ------------------
    - Chunk calculation logic: Scene length is hardcoded to be the maximum between the 
      required `context_needed` and the shortest file length in the dataset.
    """

    context_needed = hist_length * step_size

    len_files_list = [get_npy_file_shape(f)[0] for f in cls_files]

    min_len_file = np.min(len_files_list)

    # Length of the scenes
    scenes_len = max(context_needed, min_len_file)

    indices_scenes = []

    for file_num, seq_len in enumerate(len_files_list):

        for start in range(0, seq_len, scenes_len):

            stop = start+scenes_len
            
            if stop > seq_len:
                stop = seq_len

                if (stop-start) < context_needed:
                    break

            indices_scenes.append((file_num, start, stop))
    
    return indices_scenes
        


            
def load_transition_data_from_scene(file_path, scene_indices, hist_length, step_size):

    """
    Loads strided sliding-window transitions from specific dataset scenes using memory mapping.
    
    This function samples scenes with replacement to construct a training batch. It uses 
    strided slicing (Method 2) to maintain maximum data efficiency without decimating the dataset.

    Parameters
    ----------
    file_path : pathlib.Path
        The directory path containing the `*sequence*.npy` and `*CH4*.npy` files.
    scene_indices : list of tuples
        A list mapping valid scenes, formatted as `(file_num, start_index, stop_index)`.
    hist_length : int
        The number of historical measurements required for the model input.
    step_size : int
        The strided gap between consecutive measurements within the history window.

    Returns
    -------
    z_list : np.ndarray
        The context cls tokens. Shape: (Total_Samples, hist_length, 384).
    a_list : np.ndarray
        The context actions. Shape: (Total_Samples, hist_length, 1).
    y_list : np.ndarray
        The target cls tokens to be predicted. Shape: (Total_Samples, 384).

    Hardcoded Elements
    ------------------
    - Sampling strategy: Hardcoded to sample `len(scene_indices)` items with replacement 
      using a uniform distribution (`torch.randint`).
    - File pairing logic: Assumes `cls_files` and `CH4_files` sort identically and their 
      trailing numbers in the filename strictly match.
    - Latent dimension: Assumed to be 384 based on the DINOv2 embeddings shape.
    - Action dimension: Assumed to be 1 (CH4 flow).
    - File matching: Hardcoded to glob for "*sequence*.npy" and "*CH4*.npy" files.
    """

    cls_files = sorted(file_path.glob("*sequence*.npy"), key=(lambda p: int(p.stem.split('_')[-1])))
    CH4_files = sorted(file_path.glob("*CH4*.npy"), key=(lambda p: int(p.stem.split('_')[-1])))

    #Safety check
    if len(cls_files) != len(CH4_files):
         raise ValueError(f"File count mismatch: {len(cls_files)} sequence files vs {len(CH4_files)} CH4 files.")

    z_list, a_list, y_list = [], [], []
    
    #Sampling with replacement
    num_scenes = len(scene_indices)
    chosen_scenes = torch.randint(0, num_scenes, (num_scenes,)).tolist()


    for s in chosen_scenes:

        file_num, start, stop = scene_indices[s]
        
        cls_file = cls_files[file_num]
        CH4_file = CH4_files[file_num]

        #Safety check
        if cls_file.stem.split('_')[-1] != CH4_file.stem.split('_')[-1]:
             raise ValueError(f"Pairing mismatch: {cls_file.name} with {CH4_file.name}")

        # Use memory mapping to slice directly from disk.
        scene_cls = np.load(cls_file, mmap_mode='r')[start:stop]
        scene_CH4 = np.load(CH4_file, mmap_mode='r')[start:stop]

        N = scene_cls.shape[0]

        #Safety check
        if N != scene_CH4.shape[0]:
            raise ValueError(f"Length mismatch between {cls_file} and {CH4_file}. Probably they are not synchronized")
        
        context = (hist_length * step_size)
        max_idx = N -  context
        
        # Make sliding window 
        for i in range(max_idx):
            
            # Slice with a step! [start : stop : step]
            z_0 = scene_cls[i : i + context : step_size]
            a_0 = scene_CH4[i : i + context : step_size]

            # The target is the next step_size jump after the history ends
            z_1 = scene_cls[i + context]

            z_list.append(z_0)
            a_list.append(a_0)
            y_list.append(z_1)

    return np.array(z_list), np.array(a_list), np.array(y_list)
