import torch
from pathlib import Path
from src.utils.logger import generate_image_from_log
from src.data_handling.hdf5_processor import HDF5Processor
from src.models import DinoEncoder
import argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate a video replay of the autonomous growth process from log files.")
    parser.add_argument('--log_name', type=str, required=True, help="Name of the log file (without .csv extension) to generate the video from.")
    parser.add_argument('--movie_num', type=int, required=True, help="Movie number to use for frame and flow data.")
    args = parser.parse_args()
    # Standard setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_processor = HDF5Processor(encoder=DinoEncoder())
    
    movie_num = args.movie_num
    log_name = args.log_name

    log_path = Path(f"/data/lmcat/Computer_vision/automated_graphene_growth/logs/{log_name}.csv")
    output_img_path = Path(f"/data/lmcat/Computer_vision/automated_graphene_growth/plots/summary_{log_name+'_'+str(movie_num) if log_name=='validation' else log_name}.png")


    indices_frames_to_process = [30, 60, 90, 120, -1]
    target_frame_movie_num = 7
    target_frame_idx = 150
    
    
    target_frame = data_processor.get_frame_data(target_frame_movie_num, target_frame_idx)

    # Create video frames from logs
    generate_image_from_log(csv_log_path=log_path,
                             movie_num=movie_num,
                             indices_frames_to_process=indices_frames_to_process,
                             target_frame=target_frame,
                             data_processor=data_processor,
                             save_path=output_img_path)