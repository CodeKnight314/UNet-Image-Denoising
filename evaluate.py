from model import UNet
from dataset import load_dataset
from loss import PSNR, SSIM

import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def evaluate(model: torch.nn.Module, eval_dl: DataLoader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    psnr_loss = PSNR()
    ssim_loss = SSIM()

    total_psnr_loss = 0.0
    total_ssim_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(tqdm(eval_dl, desc="[Evaluating Model]")):
            clean_img, noisy_img = data
            clean_img, noisy_img = clean_img.to(device), noisy_img.to(device)
            prediction = model(noisy_img)

            psnr = psnr_loss(clean_img, prediction)
            ssim = ssim_loss(clean_img, prediction)

            total_psnr_loss += psnr.item()
            total_ssim_loss += ssim.item()

    avg_psnr_loss = total_psnr_loss / len(eval_dl)
    avg_ssim_loss = total_ssim_loss / len(eval_dl)

    print(f"Average PSNR: {avg_psnr_loss:.4f}")
    print(f"Average SSIM: {avg_ssim_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing evaluation data")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to the saved model weights")
    
    args = parser.parse_args()
    
    val_dl = load_dataset(args.data_dir, patch_size=384, batch_size=16, mode="val")
    model = UNet()
    model.load_state_dict(torch.load(args.model_weights))

    evaluate(model, val_dl)