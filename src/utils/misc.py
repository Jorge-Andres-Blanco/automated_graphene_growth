import shutil
from pathlib import Path
import imageio.v3 as iio
import numpy as np

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