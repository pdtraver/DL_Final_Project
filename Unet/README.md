# UNet Model for Semantic Segmentation on CLEVR Dataset

This documentation covers a PyTorch implementation of the UNet model for semantic segmentation, tailored for the CLEVR dataset. The CLEVR dataset comprises synthetic images of complex scenes with multiple objects, each associated with a unique mask, facilitating the segmentation of individual objects. This implementation encompasses data loading, model definition, training, inference, and saving mechanisms.

## Requirements

To run the code, the following libraries are required:

- Python 3.x
- PyTorch
- torchvision
- NumPy
- PIL

You can install these dependencies using pip:

```bash
pip install torch torchvision numpy pillow
```

## Dataset

The CLEVR dataset should be organized in a specific directory structure. For training, each subdirectory named `videoX` should contain images named `image_Y.png` and a single `mask.npy` file containing masks for all images in that directory. For inference, the structure is similar but focuses on generating and saving new masks based on the model's predictions.

## Model

The UNet model implemented here features an encoder-decoder architecture with skip connections. The encoder downsamples the input progressively, while the decoder upsamples the feature maps back to the original image size. The final output is a tensor where each channel corresponds to the predicted mask of an object in the scene.

## Usage

### Model Training

1. **Dataset Preparation**: Ensure your CLEVR dataset is structured as required and specify the path to the dataset when initializing the `CLEVRDataset` class.

2. **Model Training**:
    - Set the hyperparameters `num_epochs`, `batch_size`, and `learning_rate` as desired.
    - Initialize the dataset and dataloader with the specified batch size and sampler.
    - Create an instance of the UNet model and move it to the appropriate device (CUDA if available).
    - Define the loss function and optimizer.
    - Run the training loop, iterating over the dataset for the specified number of epochs, computing the loss, and updating the model parameters.

3. **Checkpoint Saving**: The model's state is saved every 5 epochs to a checkpoint file named `Unet_checkpoint_epoch_X.pt`, where `X` is the epoch number.

### Model Inference and Result Saving

1. **Model Initialization**: Load the pre-trained UNet model from a checkpoint and ensure it is moved to the appropriate device (CUDA if available).

2. **Dataset and DataLoader Setup**: Initialize the `CLEVRDataset` class with the path to your dataset. Use a `DataLoader` to handle batching of data, with a `SequentialSampler` to maintain the order of data.

3. **Inference**: Run the model in inference mode to generate segmentation masks for the images in the dataset. The results are saved in a specified directory, mirroring the input data structure.

4. **Result Saving**: The segmentation masks are saved as NumPy arrays. The batch size is set to 22 to match the number of images in each folder, simplifying the process of saving the mask into the correct folder.

## Running the Code

Ensure all requirements are installed, and the dataset is prepared as described. Adjust the paths and parameters as necessary based on your setup. The implementation is designed for both educational and research purposes, and performance may vary based on dataset structure, model configuration, and hardware capabilities. Adjustments might be necessary for optimal results on different setups or for operational deployment.