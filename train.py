from model import UNet 
from dataset import load_dataset
from loss import PSNR, SSIM, MSE_Loss
from utils.log_writer import LOGWRITER 
from utils.early_stop import EarlyStopMechanism

import argparse 
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import torch 
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm 
from typing import Tuple

def train(model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, optimizer: torch.optim, logger: LOGWRITER, output_dir: str, epochs: int):
    es_mech = EarlyStopMechanism(metric_threshold=0.05, grace_threshold=10, save_path=os.path.join(output_dir, "saved_weights"))
    
    mse_loss = MSE_Loss()
    psnr_loss = PSNR() 
    ssim_loss = SSIM()
    
    scaler = GradScaler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=optimizer.defaults['lr'] / 100)
    
    for epoch in range(epochs): 
        model.train()
        total_mse_loss = 0.0
        for i, data in enumerate(tqdm(train_dl, desc=f"[Training Model][{epoch+1}/{epochs}]")): 
            clean_img, noisy_img = data 
            clean_img, noisy_img = clean_img.to(device), noisy_img.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                prediction = model(noisy_img)
                loss = mse_loss(clean_img, prediction)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_mse_loss += loss.item()
        
        scheduler.step()

        # Validation phase
        model.eval() 
        total_val_mse_loss = 0.0 
        total_psnr_loss = 0.0
        total_ssim_loss = 0.0
        
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_dl, desc=f"[Validating Model][{epoch+1}/{epochs}]")): 
                clean_img, noisy_img = data 
                clean_img, noisy_img = clean_img.to(device), noisy_img.to(device)
                prediction = model(noisy_img)
                loss = mse_loss(clean_img, prediction)
                psnr = psnr_loss(clean_img, prediction)
                ssim = ssim_loss(clean_img, prediction)
                
                total_val_mse_loss += loss.item() 
                total_psnr_loss += psnr.item()
                total_ssim_loss += ssim.item()
                
        avg_mse_loss = total_mse_loss / len(train_dl)
        avg_val_mse_loss = total_val_mse_loss / len(val_dl)
        avg_psnr_loss = total_psnr_loss / len(val_dl) 
        avg_ssim_loss = total_ssim_loss / len(val_dl)

        es_mech.step(model=model, metric=avg_val_mse_loss)    
        if es_mech.check(): 
            logger.write("[INFO] Early Stopping Mechanism Engaged. Training procedure ended early.")
            break
        
        logger.write(epoch=epoch+1, tr_loss=avg_mse_loss, val_loss=avg_val_mse_loss, psnr=avg_psnr_loss, ssim=avg_ssim_loss)
        
        # Save the best model
        if avg_val_mse_loss < es_mech.best_metric:
            torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch+1}.pth"))

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing all subsets of data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for saving model weights and logs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--path", type=str, help="Model weights to load onto UNet")
    parser.add_argument("--level", type=int, help="Single noise Level for noise generation")
    parser.add_argument("--range", type=Tuple(int), default=[15, 50], help="Range of noise level defined as two numbers, start and end bound.")
    args = parser.parse_args()
    
    # Load dataset load_dataset(root_dir: str, patch_size: int, batch_size: int, mode: str="train")
    train_dl = load_dataset(args.root_dir, patch_size=256, batch_size=16, mode="train", noise_level=args.level, noise_range=args.range)
    val_dl = load_dataset(args.root_dir, patch_size=384, batch_size=16, mode="val", noise_level=args.level, noise_range=args.range)
    
    model = UNet()
    if args.path: 
        model.load_state_dict(torch.load(args.path, weights_only=True))
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    logger = LOGWRITER(args.output_dir, args.epochs)
    
    train(model, train_dl, val_dl, optimizer, logger, args.output_dir, args.epochs)