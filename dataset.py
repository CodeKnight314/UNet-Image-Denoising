import os 
import random 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T 
from glob import glob 
from PIL import Image 

class ImageDataset(Dataset): 
    def __init__(self, clean_dir: str, degradation_dirs: str, patch_size: int, v_threshold: float, h_threshold: float, transforms=None): 
        self.clean_dir = sorted(glob(os.path.join(clean_dir, "*")))
        self.degra_dir = sorted(glob(os.path.join(degradation_dirs, "*")))
        self.patch_size = patch_size
        
        self.v_threshold = v_threshold 
        self.h_threhshold = h_threshold

        if transforms: 
            self.transforms = transforms
        else: 
            self.transforms = T.Compose([T.ToTensor()])

    def __len__(self): 
        return len(self.clean_dir)
    
    def __getitem__(self, index: int): 
        clean_img = self.transforms(Image.open(self.clean_dir[index]).convert("RGB"))
        degra_img = self.transforms(Image.open(self.degra_dir[index]).convert("RGB"))

        img_w, img_h = clean_img.shape[1], clean_img.shape[2]
        d_img_w, d_img_h = degra_img.shape[1], degra_img.shape[2]

        for dim in [img_w, img_h, d_img_h, d_img_w]: 
            if(dim < self.patch_size): 
                raise ValueError(f"[ERROR] Patch size is greater than image dimensions. \n [Error] Patch Size: {self.patch_size}. Image dimension: {dim}")
        
        start_x = random.randint(0, img_w - self.patch_size)
        start_y = random.randint(0, img_h - self.patch_size)
        clean_img = clean_img[:, start_x:(start_x + self.patch_size), start_y:(start_y + self.patch_size)]
        degra_img = degra_img[:, start_x:(start_x + self.patch_size), start_y:(start_y + self.patch_size)]

        if random.random() > self.v_threshold: 
            clean_img = T.functional.vflip(clean_img)
            degra_img = T.functional.vflip(degra_img)
        if random.random() > self.h_threhshold: 
            clean_img = T.functional.hflip(clean_img)
            degra_img = T.functional.hflip(degra_img)

        return clean_img, degra_img
    
def load_dataset(root_dir: str, patch_size: int, batch_size: int, mode: str="train"):
    """
    Helper function for defining denoising dataloader
    
    Args:
        root_dir (str): directory containing all dataset subsets 
        patch_size (int): patch size to crop from both clean and noisy image
        batch_size (int): batch size of data loader
        mode (str): specified subset of dataset for dataloader
        
    Returns: 
        torch.data.utils.DataLoader: Dataloader class of the dataset
    """
    assert mode in ["train", "val", "test"], f"[ERROR] Invalid mode for dataset. Mode {mode} is not available."
    clean_dir = os.path.join(root_dir, os.path.join(mode, "clean"))
    degraded_dir = os.path.join(root_dir, os.path.join(mode, "degraded"))
    dataset = ImageDataset(clean_dir=clean_dir, 
                        degradation_dirs=degraded_dir, 
                        patch_size=patch_size, 
                        v_threshold=0.25, 
                        h_threshold=0.25)
    
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)