import shutil
from pathlib import Path
import imageio.v3 as iio
import numpy as np
import yaml
from src.models.transition import EnsembleTransitionModel

def cleanup_directory(dir: Path):
    """
    Deletes all files in the specified directory.
    """
    if dir.exists() and dir.is_dir():
        shutil.rmtree(dir)
        print(f"Cleaned up directory: {dir}")

    else:
        print(f"Directory {dir} does not exist or is not a directory.")


def compile_video_from_frames(saved_images: list[str | Path] | None,
                              temp_dir: str | Path | None,
                              output_video_path: str | Path,
                              fps=4):

    # Compile Video with ImageIO
    print("Compiling video...")

    if saved_images is None and temp_dir is not None:
        saved_images = sorted(temp_dir.glob("frame_*.png"))


    with iio.imopen(output_video_path, "w", plugin="pyav") as out_file:
        out_file.init_video_stream("libx264", fps=fps)
        for img_path in saved_images:

            frame = iio.imread(img_path)

            if frame.ndim == 3 and frame.shape[-1]==4:
                frame = frame[:, :, :3]
            
            out_file.write_frame(frame)

    if temp_dir:
        cleanup_directory(temp_dir)
            
    print(f"Video successfully saved to {output_video_path}")

    return None

def load_yaml_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_model_from_yaml_config(config_path: Path) -> EnsembleTransitionModel:
    """
    Instantiates an EnsembleTransitionModel and loads its trained weights 
    strictly based on the parameters defined in the YAML configuration.
    """
    config = load_yaml_config(config_path)
        
    try:
        exec_cfg = config['execution']['active_model']
        model_dir = config['execution']['model_directory']
    except KeyError as e:
        raise KeyError(f"Missing required configuration block in YAML: {e}")

    # 1. Instantiate the architecture
    transition_model = EnsembleTransitionModel(
        num_models=exec_cfg['num_models'],
        latent_dim=exec_cfg['latent_dim'],
        action_dim=exec_cfg['action_dim'],
        hidden_dim=exec_cfg['hidden_dimension'],
        normalization=exec_cfg['normalization'],
        activation=exec_cfg['activation'],
        history=exec_cfg['history_size'],
        num_hidden_layers=exec_cfg['num_hidden_layers']
    )
    
    # 2. Construct the prefix exactly as expected by your training outputs
    model_name_prefix = (
        f"{model_dir}/mlp_activation_{exec_cfg['activation']}"
        f"_norm_{exec_cfg['normalization']}"
        f"_hist{exec_cfg['history_size']}"
        f"_step{exec_cfg['step_size']}"
        f"_hiddim{exec_cfg['hidden_dimension']}"
    )
    
    # 3. Load the weights
    print(f"Loading trained model weights from: {model_name_prefix}")
    transition_model.load_ensemble(model_name_prefix)
    
    # Ensure the model is set to evaluation mode (disables dropout/batchnorm updates)
    # This is a critical safety step before running inference in a physical reactor.
    transition_model.eval() 
    
    return transition_model
