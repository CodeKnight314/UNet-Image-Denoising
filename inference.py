import argparse
import os
import torch
from model import UNet
from loss import PSNR, SSIM
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Utility function to load an image
transform = transforms.Compose([
    transforms.ToTensor(),
])

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Utility function to save an image
def save_image(tensor, output_path):
    image = transforms.ToPILImage()(tensor.squeeze(0))
    image.save(output_path)

# Utility function to plot images
def plot_images(noisy_img, prediction, clean_img=None, output_path=None):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3 if clean_img is not None else 2, 1)
    plt.title("Noisy Image")
    plt.imshow(transforms.ToPILImage()(noisy_img.squeeze(0)))
    plt.axis("off")

    plt.subplot(1, 3 if clean_img is not None else 2, 2)
    plt.title("Prediction")
    plt.imshow(transforms.ToPILImage()(prediction.squeeze(0)))
    plt.axis("off")

    if clean_img is not None:
        plt.subplot(1, 3, 3)
        plt.title("Ground Truth")
        plt.imshow(transforms.ToPILImage()(clean_img.squeeze(0)))
        plt.axis("off")

    if output_path is not None:
        plt.savefig(output_path)
    plt.show()

# Inference function
def inference(model, input_path, output_dir, ground_truth_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    psnr_metric = PSNR()
    ssim_metric = SSIM()

    # Check if input is a single image or a directory
    if os.path.isdir(input_path):
        noisy_images = [os.path.join(input_path, img) for img in os.listdir(input_path)]
        ground_truths = [os.path.join(ground_truth_path, img) for img in os.listdir(input_path)] if ground_truth_path else None
    else:
        noisy_images = [input_path]
        ground_truths = [ground_truth_path] if ground_truth_path else None

    with torch.no_grad():
        for idx, noisy_image_path in enumerate(noisy_images):
            noisy_img = load_image(noisy_image_path).to(device)
            prediction = model(noisy_img)

            # Prepare output paths
            base_name = os.path.basename(noisy_image_path)
            output_image_path = os.path.join(output_dir, f"predicted_{base_name}")
            plot_output_path = os.path.join(output_dir, f"plot_{base_name}.png")

            # Save the predicted image
            save_image(prediction.cpu(), output_image_path)

            if ground_truths:
                ground_truth_img = load_image(ground_truths[idx]).to(device)
                psnr_value = psnr_metric(ground_truth_img, prediction).item()
                ssim_value = ssim_metric(ground_truth_img, prediction).item()
                print(f"Image: {base_name} | PSNR: {psnr_value:.4f} | SSIM: {ssim_value:.4f}")
                plot_images(noisy_img.cpu(), prediction.cpu(), clean_img=ground_truth_img.cpu(), output_path=plot_output_path)
            else:
                plot_images(noisy_img.cpu(), prediction.cpu(), output_path=plot_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input image or directory of images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the predicted images and plots")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to the saved model weights")
    parser.add_argument("--ground_truth_path", type=str, help="Path to the ground truth image or directory of images")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load model
    model = UNet()
    model.load_state_dict(torch.load(args.model_weights))
    
    # Run inference
    inference(model, args.input_path, args.output_dir, ground_truth_path=args.ground_truth_path)