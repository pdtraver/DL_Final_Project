# UNet Model for Semantic Segmentation on CLEVR Dataset

This contains a PyTorch implementation of a UNet model for semantic segmentation, specifically designed for the CLEVR dataset. The CLEVR dataset consists of synthetic images of complex scenes with multiple objects. Each object in an image is associated with a unique mask, allowing for the segmentation of individual objects. This implementation includes data loading, model definition, training, and saving mechanisms.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- NumPy
- PIL

## Dataset

The CLEVR dataset should be organized in a directory structure where each subdirectory named `videoX` contains images named `image_Y.png` and a single `mask.npy` file containing masks for all images in that directory. The dataset path should be specified when initializing the `CLEVRDataset` class.

## Model

The UNet model implemented here consists of an encoder-decoder architecture with skip connections. The encoder progressively downsamples the input, while the decoder upsamples the feature maps to the original image size. The final output is a tensor where each channel corresponds to the predicted mask of an object in the scene.

## Usage

1. **Dataset Preparation**: Ensure your CLEVR dataset is structured as required and specify the path to the dataset when initializing the `CLEVRDataset` class.

2. **Model Training**:
    - Set the hyperparameters `num_epochs`, `batch_size`, and `learning_rate` as desired.
    - Initialize the dataset and dataloader with the specified batch size and sampler.
    - Create an instance of the UNet model and move it to the appropriate device (CUDA if available).
    - Define the loss function and optimizer.
    - Run the training loop, which iterates over the dataset for the specified number of epochs, computes the loss, and updates the model parameters.

3. **Checkpoint Saving**: The model's state is saved every 5 epochs to a checkpoint file named `Unet_checkpoint_epoch_X.pt`, where `X` is the epoch number.

## Code Structure

- **Device Configuration**: Checks for CUDA availability and sets the default device accordingly.
- **CLEVRDataset Class**: Custom PyTorch dataset class for loading the CLEVR dataset. It includes methods for preprocessing images and masks.
- **UNet Class**: Defines the UNet model architecture.
- **Training Loop**: Iterates over the dataset, computes the loss, updates the model parameters, and saves checkpoints.

## Running the Code

Ensure all requirements are installed and the dataset is prepared as described. You can then run the script to start training the model. Adjust the hyperparameters as needed for your specific setup or requirements.