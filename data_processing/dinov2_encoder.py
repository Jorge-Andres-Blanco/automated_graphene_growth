import torch
import torchvision.transforms as T
import numpy as np


class DinoEncoder:
    
    def __init__(self, model_name='dinov2_vits14_reg', device=None):
        """
        Initializes the DINOv2 encoder, loads the weights, and sets up the transformation pipeline.
        """

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Loading {model_name} on {self.device}...")

        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.to(self.device)
        self.model.eval()

        # Transformation and Normalization
        self.transform = T.Compose([
            T.CenterCrop(1498),
            
            # Downsized to approximately half
            T.Resize(742), 
            
            # Apply DINOv2's expected statistical distribution
            #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    
    
    @torch.no_grad()
    def encode_numpy_array(self, images_array: np.ndarray, batch_size: int = 16, save_file_name: str = None) -> np.ndarray:
        """
        Processes an NumPy array of shape (N, Height, Width).
        Automatically handles scaling, channel expansion, and batched GPU extraction.
        """
        if images_array.ndim != 3:
            raise ValueError(f"Expected array of shape (N, H, W), got {images_array.shape}")

        N = images_array.shape[0]
        all_embeddings = []

        # Process in chunks
        for i in range(0, N, batch_size):

            # Slice the batch
            batch_np = images_array[i : i + batch_size]

            # Scale to [0.0, 1.0]
            batch_np = batch_np.astype(np.float32) / 4095.0

            # Expand to 3 Channels: (B, 3, 1540, 2056) so that DINO can read
            batch_np = np.repeat(batch_np[:, np.newaxis, :, :], 3, axis=1)

            batch_tensor = torch.from_numpy(batch_np).to(self.device)

            # Downsample to (B, 3, 742, 742)
            batch_tensor = self.transform(batch_tensor)

            # Get embeddings
            embeddings = self.model(batch_tensor)
            
            all_embeddings.append(embeddings.cpu().numpy())


            print(f"Processed batch {i // batch_size + 1} / {int(np.ceil(N / batch_size))}")

        
        # Stack into a single matrix (N, 384)
        embeddings_array = np.vstack(all_embeddings)

        if save_file_name is not None:
            np.save(file=save_file_name, arr=embeddings_array)
        else:
            np.save(file="embeddings", arr=embeddings_array)

        return embeddings_array
        