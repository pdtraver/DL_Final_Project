# Semantic Segmentation Prediction Model

This documentation covers an implementation using SimVP and UNet models for semantic segmentation and mask prediciton, tailored for our dataset. The dataset comprises synthetic images of complex scenes with multiple objects, each associated with a unique mask, facilitating the segmentation of individual objects. This implementation encompasses the inference stage. 

# Requirements

To run the code, you will need to clone the public repo OpenSTL: https://github.com/chengtan9907/OpenSTL.git. To do so you can run the following:
~/SimVP/installSimVP.sh. 

This will clone the repo and install of the requirements. 

You can install all other dependencies using pip.

## Running the Code

Run dataset/dataloaders.py to get the dataset in the required format. The directories and paths need to be changed accordingly. 
Run predictHidden.py to get the final predicted masks. Again, you will need to change the paths for the loaded data.
