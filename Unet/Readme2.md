# UNet Model for Image Segmentation on CLEVR Dataset

This repository contains a PyTorch implementation of the UNet model for image segmentation, tailored for use with the CLEVR dataset. The code includes mechanisms for loading a pre-trained model, performing inference, and saving the segmentation results. This implementation is particularly useful for processing and analyzing synthetic images from the CLEVR dataset to understand object segmentation performance.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- NumPy
- PIL

## Setup

Ensure that all required libraries are installed. You can install them using pip:

```bash
pip install torch torchvision numpy pillow
```

## Dataset Structure

The CLEVR dataset should be organized in directories where each `videoX` directory contains images named `image_Y.png`. The dataset path should be specified when initializing the `CLEVRDataset` class.

## Model Overview

The UNet model implemented here is designed for semantic segmentation tasks. It features an encoder-decoder architecture with skip connections, which helps in recovering spatial hierarchies between low-level and high-level features. The model is capable of outputting a segmentation mask for each object in the image.

## Usage

1. **Model Initialization**: Load the pre-trained UNet model from a checkpoint. Ensure the model is moved to the appropriate device (CUDA if available).

2. **Dataset and DataLoader Setup**: Initialize the `CLEVRDataset` class with the path to your dataset. Use a `DataLoader` to handle batching of data.

3. **Inference**: Run the model in inference mode to generate segmentation masks for the images in the dataset. The results are saved in a specified directory.

4. **Result Saving**: The segmentation masks are saved as NumPy arrays in a directory structure that mirrors the input data.

## Detailed Steps

1. **Load the Model**:
    - Ensure the checkpoint file path is correctly specified.
    - Load the model weights from the checkpoint.
    - Set the model to evaluation mode using `model.eval()`.

2. **Prepare the DataLoader**:
    - Initialize the `CLEVRDataset`.
    - Use a `SequentialSampler` to maintain the order of data.
    - Specify the batch size (should match the number of images in each folder for simplicity).

3. **Run Inference and Save Results**:
    - For each batch of images, compute the output masks using the model.
    - Convert the output tensors to NumPy arrays and save them in the corresponding directory.

## Running the Code

To execute the script, ensure that your dataset is correctly placed and the model checkpoint is accessible. Run the script using a Python interpreter. Adjust the paths and parameters as necessary based on your setup. Here the batch size is set to 22 because for a single folder, there are 22 files. This makes it easy to save the mask into the right folder

## Note

This implementation is intended for educational and research purposes. The performance and efficiency can vary based on the dataset structure, model configuration, and hardware capabilities. Adjustments might be necessary for optimal results on different setups or for operational deployment.