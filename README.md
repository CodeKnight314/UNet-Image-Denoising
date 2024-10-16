# UNet-Image-Denoising

## Overview
This is an educational repository for training a standard U-Net model for image denoising. The project supports training with either a consistent noise level or varying noise levels across different data samples. The specific noise configuration can be toggled based on command-line arguments. If a specific noise level is not provided, the training script defaults to using a noise range and augments data accordingly.

Training samples are randomly cropped and resized to 256x256, while validation samples are cropped and resized to 384x384. The dataset used for training and validation is sourced from COCO, with images smaller than the required dimensions filtered out.

## Visual Results
The UNet model was trained on a uniform distribution of noise levels ranging from 15 to 50, with a learning rate of 0.0001 for 25 epochs. Below are some visual results from the COCO validation set.

<p align="center">
  <img src="visuals/sample_1.png" alt="Sample 1">
  <br>
  <em>Fairly faithful restoration but with detail loss</em>
</p>
<p align="center">
  <img src="visuals/sample_2.png" alt="Sample 2">
  <br>
  <em>Faithful restoration but with color difference</em>
</p>
<p align="center">
  <img src="visuals/sample_3.png" alt="Sample 3">
  <br>
  <em>Fairly faithful restoration but with detail loss</em>
</p>

As seen in the visual samples, the UNet model can perform general image denoising but struggles with fine details, possibly due to needing to learn from multiple noise levels simultaneously. However, with modifications such as incorporating a noise level predictor, the model could likely achieve better results. Despite this, the U-Net still produces acceptable results with minimal structural changes.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/CodeKnight314/UNet-Image-Denoising.git
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv UNet-env
    source UNet-env/bin/activate
    ```

3. Change to the project directory:
    ```bash
    cd UNet-Image-Denoising/
    ```

4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Training
To train the model, use the `train.py` script with the following arguments:

- `--root_dir` (required): Root directory containing all subsets of data.
- `--output_dir` (required): Directory for saving model weights and logs.
- `--lr` (default: 1e-4): Learning rate for the model.
- `--epochs` (default: 100): Number of epochs for training.
- `--path`: Path to model weights to load onto UNet (optional).
- `--level`: Single noise level for noise generation (optional).
- `--range` (default: `[15, 50]`): Range of noise levels, defined as two numbers representing the start and end bounds.

Example usage:
```bash
python train.py --root_dir /path/to/data --output_dir /path/to/save --lr 1e-4 --epochs 25 --range 15 50
```

## Inference
To perform inference on an image or a directory of images, use the `inference.py` script with the following arguments:

- `--input_path` (required): Path to the input image or directory of images.
- `--output_dir` (required): Directory to save the predicted images and plots.
- `--model_weights` (required): Path to the saved model weights.
- `--ground_truth_path`: Path to the ground truth image or directory of images (optional).

Example usage:
```bash
python inference.py --input_path /path/to/images --output_dir /path/to/save --model_weights /path/to/weights
```

## Evaluation
To evaluate the model on a dataset, use the `evaluate.py` script with the following arguments:

- `--data_dir` (required): Directory containing evaluation data.
- `--model_weights` (required): Path to the saved model weights.

Example usage:
```bash
python evaluate.py --data_dir /path/to/eval_data --model_weights /path/to/weights