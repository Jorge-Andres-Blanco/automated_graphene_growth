import torch
import torchvision.transforms as T
import numpy as np
import ssl
from sklearn.decomposition import PCA

# Bypass institutional SSL interception xd
ssl._create_default_https_context = ssl._create_unverified_context


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
    def encode_numpy_array(self, images_array: np.ndarray, batch_size: int = 16, save_file_name: str = None, verbose: bool = False) -> np.ndarray:
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


            if verbose:
                print(f"Processed batch {i // batch_size + 1} / {int(np.ceil(N / batch_size))}")

        
        # Stack into a single matrix (N, 384)
        embeddings_array = np.vstack(all_embeddings)

        return embeddings_array


    @torch.no_grad()
    def get_attention_map(self, image: np.ndarray, save_path: str = None):
        """
        Returns the self-attention map of the key tokens over the image. The attention map is averaged over all heads.
        Parameters:
            image (np.ndarray): Input image of shape (H, W) or (H, W, 3).
            save_path (str): Optional path to save the attention visualization.
        """

        # Grayscale to RGB
        if image.ndim == 2:
            image = np.repeat(image[np.newaxis, :, :], 3, axis=0)

        # Normalize
        image = image.astype(np.float32) / 4095.0

        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)

        # Downsample to (1, 3, 742, 742)
        image_tensor = self.transform(image_tensor)

        # Get dimentions of attention map
        h_featmap = image_tensor.shape[2] // 14
        w_featmap = image_tensor.shape[3] // 14


        # Extract attention maps from the last layer
        # Returns shape (batch_size (1), Heads (6), Tokens (Queries, (742/14)**2+1), Tokens (Keys, (742/14)**2+1))
        attention_maps = self.model.get_last_selfattention(image_tensor)

        # Take all heads (to avg), then take the keys (removing the CLS token)
        cls_attention = attention_maps[0, :, 0, 1:].mean(dim=0)  # Shape: (Heads, Tokens)

        cls_attention_map = cls_attention.reshape(h_featmap, w_featmap).cpu().numpy()

        return cls_attention_map
    

    @torch.no_grad()
    def fit_pca_to_images(self, images_array: torch.Tensor) -> PCA:
        """
        Fits a PCA model to the embeddings of the provided images.
        Parameters:
            images_array (torch.Tensor): Input images of shape (N, 3, H, W).
        Returns:
            PCA: Fitted PCA model.
        """

        # Get features from the model
        features = self.model.forward_features(images_array)

        # Patch embeddings: (N, Tokens (742/14)**2, Embedding_dim (384))
        patch_tokens = features['x_norm_patchtokens']  # Shape: (N, Tokens, Embedding_dim)

        patch_tokens_np = patch_tokens.cpu().numpy()
        patch_tokens_np = patch_tokens_np.reshape(-1, patch_tokens_np.shape[-1]) # Reshape to (N*Tokens, Embedding_dim)

        pca = PCA(n_components=3)

        pca.fit(patch_tokens_np)  # Reshape to (N*Tokens, Embedding_dim)

        return pca
    
    @torch.no_grad()
    def visualize_pca(self, image_array: np.ndarray):
        """
        Transforms images into a 3-channel image for visualization based on the main 3 PCA  components.
        Do not use with many images (limit to less than 10)
        Parameters:
            image_array (np.ndarray): Set of images to perform the PCA and visualization (N, H, W).
        """
        imgs_np = image_array.astype(np.float32) / 4095.0
        imgs_np = np.repeat(imgs_np[:, np.newaxis, :, :], 3, axis=1)  # Expand to 3 channels
        imgs_tensor = torch.from_numpy(imgs_np).to(self.device)
        imgs_tensor = self.transform(imgs_tensor)

        # Fit PCA on the entire batch of images
        fitted_pca = self.fit_pca_to_images(imgs_tensor)

        h_featmap = imgs_tensor.shape[2] // 14
        w_featmap = imgs_tensor.shape[3] // 14

        features = self.model.forward_features(imgs_tensor)
        patch_tokens = features['x_norm_patchtokens']

        # Calculate bounds across tokens to preserve color meaning
        all_tokens_flat = patch_tokens.cpu().numpy().reshape(-1, 384)
        all_pca_flat = fitted_pca.transform(all_tokens_flat)
        global_min = all_pca_flat.min(axis=0)
        global_max = all_pca_flat.max(axis=0)

        pca_transformed_images = np.zeros((imgs_tensor.shape[0], h_featmap, w_featmap, 3))
        for i in range(patch_tokens.shape[0]):
            pca_features = fitted_pca.transform(patch_tokens[i].cpu().numpy())
            pca_features = (pca_features - global_min) / (global_max - global_min)
            
            pca_transformed_images[i] = pca_features.reshape(h_featmap, w_featmap, 3)

        return pca_transformed_images