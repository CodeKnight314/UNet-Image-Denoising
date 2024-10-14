import os 
import random 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T 
from glob import glob 
from PIL import Image 

class ImageDataset(Dataset): 
    def __init__(self, clean_dir: str, patch_size: int, noise_level: float = None, noise_range: tuple = None, v_threshold: float = 0.5, h_threshold: float = 0.5, transforms=None): 
        self.clean_dir = sorted(glob(os.path.join(clean_dir, "*")))
        self.patch_size = patch_size
        self.noise_level = noise_level
        self.noise_range = noise_range
        
        self.v_threshold = v_threshold 
        self.h_threshold = h_threshold

        if transforms: 
            self.transforms = transforms
        else: 
            self.transforms = T.Compose([T.ToTensor()])

    def __len__(self): 
        return len(self.clean_dir)
    
    def __getitem__(self, index: int): 
        clean_img = self.transforms(Image.open(self.clean_dir[index]).convert("RGB"))

        img_w, img_h = clean_img.shape[1], clean_img.shape[2]

        if img_w < self.patch_size or img_h < self.patch_size: 
            raise ValueError(f"[ERROR] Patch size is greater than image dimensions. \n [Error] Patch Size: {self.patch_size}. Image dimensions: ({img_w}, {img_h})")
        
        # Extract a random patch
        start_x = random.randint(0, img_w - self.patch_size)
        start_y = random.randint(0, img_h - self.patch_size)
        clean_img = clean_img[:, start_x:(start_x + self.patch_size), start_y:(start_y + self.patch_size)]
        
        # Add noise to the clean image
        if self.noise_level is not None:
            noise = torch.randn_like(clean_img) * (self.noise_level / 255.0)
        elif self.noise_range is not None:
            noise_level = random.uniform(self.noise_range[0], self.noise_range[1])
            noise = torch.randn_like(clean_img) * (noise_level / 255.0)
        else:
            noise = torch.zeros_like(clean_img)
        
        noisy_img = torch.clamp(clean_img + noise, 0.0, 1.0)

        # Apply random flips
        if random.random() > self.v_threshold: 
            clean_img = T.functional.vflip(clean_img)
            noisy_img = T.functional.vflip(noisy_img)
        if random.random() > self.h_threshold: 
            clean_img = T.functional.hflip(clean_img)
            noisy_img = T.functional.hflip(noisy_img)

        return clean_img, noisy_img
    
def load_dataset(root_dir: str, patch_size: int, batch_size: int, mode: str="train", noise_level: float = None, noise_range: tuple = None):
    """
    Helper function for defining denoising dataloader
    
    Args:
        root_dir (str): directory containing all dataset subsets 
        patch_size (int): patch size to crop from both clean and noisy image
        batch_size (int): batch size of data loader
        mode (str): specified subset of dataset for dataloader
        noise_level (float, optional): fixed noise level to be added to images
        noise_range (tuple, optional): range of noise levels to add random noise
        
    Returns: 
        torch.data.utils.DataLoader: Dataloader class of the dataset
    """
    assert mode in ["train", "val", "test"], f"[ERROR] Invalid mode for dataset. Mode {mode} is not available."
    clean_dir = os.path.join(root_dir, os.path.join(mode, "clean"))
    
    dataset = ImageDataset(clean_dir=clean_dir, 
                           patch_size=patch_size, 
                           noise_level=noise_level, 
                           noise_range=noise_range, 
                           v_threshold=0.25, 
                           h_threshold=0.25)
    
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)