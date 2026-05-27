import numpy as np
import torch
from pathlib import Path

class TransitionDataLoader:
    """
    Loads transition data from sequence and action .npy files into sliding windows.
    """
    
    
    def __init__(self, folder_path, step_size, hist_length):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.folder_path = Path(folder_path)
        self.step_size = step_size
        self.hist_length = hist_length
        self.context_needed = hist_length * step_size
        self.cls_files = sorted(self.folder_path.glob("*sequence*.npy"), key=lambda p: int(p.stem.split('_')[-1]))
        self.CH4_files = sorted(self.folder_path.glob("*CH4*.npy"), key=lambda p: int(p.stem.split('_')[-1]))
        self.scene_indices = self.generate_scene_indices()
    
    def _get_npy_file_shape(self, file_path):
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
        """
        
        return np.load(file_path, mmap_mode='r').shape

    
    
    def load_full_dataset(self, return_indices=False):
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
        """

        if len(self.cls_files) != len(self.CH4_files):
             raise ValueError("Mismatch between sequence and CH4 files.")

        z_list, a_list, y_list, indices = [], [], [], []
        current_index = 0

        for cls_file, CH4_file in zip(self.cls_files, self.CH4_files):
            data_cls = np.load(cls_file, mmap_mode='r')
            data_CH4 = np.load(CH4_file, mmap_mode='r')

            N = data_cls.shape[0]

            max_idx = N - self.context_needed
            
            for i in range(max_idx):

                z_list.append(data_cls[i : i + self.context_needed : self.step_size])
                a_list.append(data_CH4[i : i + self.context_needed : self.step_size])
                y_list.append(data_cls[i + self.context_needed])

            indices.append((current_index, current_index + (N - self.hist_length)))
            current_index = indices[-1][1]

        if return_indices:
            return np.array(z_list), np.array(a_list), np.array(y_list), indices
        
        return np.array(z_list), np.array(a_list), np.array(y_list)

    
    
    def generate_scene_indices(self):
        """
        Generates dataset sequence indices to split variable-length files into uniform scenes.
        
        Files are divided into chunks of length `scenes_len` (determined by the shortest 
        file in the dataset). Any remainder chunk at the end of a file is included only 
        if it satisfies the `context_needed` minimum length.

        Returns
        -------
        list of tuples
            A list of indices formatted as `(file_num, start_index, stop_index)` mapping
            valid sequence chunks for the dataloader.
        """
        len_files_list = [self._get_npy_file_shape(f)[0] for f in self.cls_files]
        
        min_len_file = np.min(len_files_list)
        scenes_len = max(self.context_needed, min_len_file)
        indices_scenes = []

        for file_num, seq_len in enumerate(len_files_list):

            for start in range(0, seq_len, scenes_len):
                
                stop = min(start + scenes_len, seq_len)

                if (stop - start) >= self.context_needed:
                    indices_scenes.append((file_num, start, stop))
                    
        return indices_scenes

    
    
    def load_from_sample_scenes_with_replacement(self, scene_indices):
        """
        Loads strided sliding-window transitions from specific dataset scenes using memory mapping.
        This function samples scenes with replacement to construct a training batch.
        
        Parameters
        ----------
        scene_indices : list of tuples
            A list mapping valid scenes, formatted as `(file_num, start_index, stop_index)`.

        Returns
        -------
        z_list : np.ndarray
            The context cls tokens. Shape: (Total_Samples, hist_length, 384).
        a_list : np.ndarray
            The context actions. Shape: (Total_Samples, hist_length, 1).
        y_list : np.ndarray
            The target cls tokens to be predicted. Shape: (Total_Samples, 384).
        """
        z_list, a_list, y_list = [], [], []
        num_scenes = len(scene_indices)
        chosen_scenes = torch.randint(0, num_scenes, (num_scenes,)).tolist()

        for s in chosen_scenes:
            file_num, start, stop = scene_indices[s]
            
            scene_cls = np.load(self.cls_files[file_num], mmap_mode='r')[start:stop]
            scene_CH4 = np.load(self.CH4_files[file_num], mmap_mode='r')[start:stop]

            N = scene_cls.shape[0]
            max_idx = N - self.context_needed
            
            for i in range(max_idx):
                z_list.append(scene_cls[i : i + self.context_needed : self.step_size])
                a_list.append(scene_CH4[i : i + self.context_needed : self.step_size])
                y_list.append(scene_cls[i + self.context_needed])

        return np.array(z_list), np.array(a_list), np.array(y_list)