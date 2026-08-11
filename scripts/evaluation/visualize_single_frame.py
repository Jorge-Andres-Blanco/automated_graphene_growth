from pathlib import Path
import h5py
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.data_handling.hdf5_processor import HDF5Processor
from src.models import DinoEncoder
from src.utils.plotting import adjust_exposure_gray_image, add_scalebar_to_ax

PROJECT_ROOT = Path(__file__).resolve().parents[2]

train_data_path = PROJECT_ROOT / "data_processing" / "training_data"
validation_data_path = PROJECT_ROOT / "data_processing" / "validation_data"


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    data_processor = HDF5Processor(encoder=DinoEncoder())

    movie_num = 1
    frame_idx = 2650

    frame_0 = data_processor.get_frame_data(movie_num, frame_idx)

    # 4. Create the Plot
    fig, axes = plt.subplots(1, 1, figsize=(4, 4), dpi=300)

    # Improve contrast:
    frame_0 = data_processor.encoder.transform(torch.tensor(frame_0, dtype=torch.float32).unsqueeze(0).to(device)).squeeze().cpu().numpy()
    frame_0 = adjust_exposure_gray_image(frame_0)

    axes.imshow(frame_0, cmap='gray')
    add_scalebar_to_ax(axes, pixels_length=143, scalebar_length=400, unit=r'$\mu$m', loc='lower right', fontsize=20)
    axes.axis('off') 


    # Encoding
    save_path = f"/data/lmcat/Computer_vision/automated_graphene_growth/images_single_frames/movie{movie_num}_frame{frame_idx}.png"

    fig.tight_layout()
    plt.savefig(save_path)

    return None

if __name__ == "__main__":
    main()