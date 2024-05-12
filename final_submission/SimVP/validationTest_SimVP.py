import numpy as np
from tqdm import tqdm
import pickle
from OpenSTL.openstl.api import BaseExperiment
from OpenSTL.openstl.utils import create_parser, default_parser
from OpenSTL.openstl.methods import method_maps
import torch
import os
import datetime
import time
import sys
import cv2

## Load validation set
def openValSet(directory):
    with open(directory + 'val.pkl', 'rb') as f:
            val = pickle.load(f)
    X_val, Y_val, X_val_mask, Y_val_mask = val['X_val'], val['Y_val'], val['X_val_mask'], val['Y_val_mask']
    print('X_val shape: ' + str(np.shape(X_val)))
    print('Y_val shape: ' + str(np.shape(Y_val)))
    print('X_val_mask shape: ' + str(np.shape(X_val_mask)))
    print('Y_test_mask shape: ' + str(np.shape(X_val_mask)))
    
    return X_val, Y_val, X_val_mask, Y_val_mask

## Get SimVP configs
def getConfigs():
    custom_training_config = {
        'pre_seq_length': 11,
        'aft_seq_length': 11,
        'total_length': 22,
        'batch_size': 16,
        'val_batch_size': 16,
        'epoch': 100,
        'lr': 0.001,   
        'metrics': ['mse', 'mae'],
        
        'ex_name': 'image_exp',
        'dataname': 'images',
        'in_shape': [11, 3, 160, 240],
    }

    custom_model_config = {
        # For MetaVP models, the most important hyperparameters are: 
        # N_S, N_T, hid_S, hid_T, model_type
        'method': 'SimVP',
        # Users can either using a config file or directly set these hyperparameters 
        # 'config_file': 'configs/custom/example_model.py',
        
        # Here, we directly set these parameters
        'model_type': 'gSTA',
        'N_S': 4,
        'N_T': 8,
        'hid_S': 64,
        'hid_T': 256
    }

    custom_training_config_mask = {
        'pre_seq_length': 11,
        'aft_seq_length': 11,
        'total_length': 2,
        'batch_size': 1,
        'val_batch_size': 1,
        'epoch': 20,
        'lr': 0.001,   
        'metrics': ['mse', 'mae'],
        
        'ex_name': 'mask_exp',
        'dataname': 'masks',
        'in_shape': [11, 1, 160, 240],
    }

    custom_model_config_mask = {
        # For MetaVP models, the most important hyperparameters are: 
        # N_S, N_T, hid_S, hid_T, model_type
        'method': 'SimVP',
        # Users can either using a config file or directly set these hyperparameters 
        # 'config_file': 'configs/custom/example_model.py',
        
        # Here, we directly set these parameters
        'model_type': 'gSTA',
        'N_S': 4,
        'N_T': 8,
        'hid_S': 64,
        'hid_T': 256
    }
    return custom_training_config, custom_model_config, custom_model_config_mask, custom_model_config_mask

## Load SimVp checkpoint model
def getSimVPModel(device, training_config, model_config, ckpt='/scratch/pdt9929/DL_Final_Project/work_dirs/image_exp/checkpoints/best.ckpt'):
    args = create_parser().parse_args([])
    config = args.__dict__
    
    # update the training config
    config.update(training_config)
    # update the model config
    config.update(model_config)
    # fulfill with default values
    default_values = default_parser()
    for attribute in default_values.keys():
        if config[attribute] is None:
            config[attribute] = default_values[attribute]
    
    save_dir = 'dataset/generated'
    model = method_maps['simvp'](save_dir=save_dir, **config)
    ckpt = torch.load(ckpt, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    return model
    
## Predict SimVP images
def predictSimVP(device, model, data, batch_size=8, save_file='dataset/val_predictions.npy'):
    # Set model to eval & send to device
    model.eval()
    model.to(device)
    
    # Convert data to tensor
    data = torch.tensor(data)
    
    # Initialize predictions with empty tensor (removed later)
    predictions = torch.zeros((1, 11, 3, 160, 240))
    
    last = 0
    # NOTE: when working with hidden set, might need to split it into two groups for .npy memory configuration
    for i in tqdm(range(int(np.ceil(data.shape[0]/batch_size)))):
        # make batch
        if (i+1)*batch_size >= data.shape[0]:
            batch = data[i*batch_size:]
        else:
            batch = data[i*batch_size:(i+1)*batch_size]
        video_tensors = batch.to(device)
        
        # predict batch
        with torch.no_grad():
            pred_frames = model(video_tensors)
        
        # unload GPU
        predictions = torch.cat((predictions, pred_frames.to('cpu')), 0)
        del pred_frames
        del video_tensors
        torch.cuda.empty_cache()
        last = i
    
    # save predictions (removing empty first tensor)
    val_predictions = predictions[1:].cpu().detach().numpy()
    with open(save_file, 'wb') as f:
        np.save(f, val_predictions)
        
    print('Validation Images Shape: ' + str(np.shape(val_predictions)))
        
    return val_predictions, save_file
  
# Generate image files from predictions
def generateImages(predictions, output_dir = 'dataset/val_predictions'):
    preds = predictions.transpose(0, 1, 3, 4, 2)
    print('Predictions Size: ' + str(preds.shape))

    # Generate images for predicted frames
    for idx, video in enumerate(tqdm(preds)):
        video_dir = f'{output_dir}/video_0{idx + 1000}'
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        for idx, image in enumerate(video):
            image_filename = f'{video_dir}/image_{idx + 11}.png'
            if not cv2.imwrite(image_filename, image):
                print(f'Could not write image {image_filename}')
            else:
                cv2.imwrite(image_filename, image*255)  

#TODO
## Transform predicted masks back to normal configuration
## Measure Jacard distance using provided code

## Main
def main(build_label_set=True):
    # get data
    print('>'*35 + ' Loading Validation Set ' + '<'*35)
    X_val, Y_val, X_val_mask, Y_val_mask = openValSet("dataset/")
    print('>'*35 + ' Val Set Loaded ' + '<'*35)
    
    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('>'*35 + ' Device Loaded ' + '<'*35)
    
    # get SimVP
    custom_training_config, custom_model_config, custom_model_config_mask, custom_model_config_mask = getConfigs()
    VPmodel = getSimVPModel(device, custom_training_config, custom_model_config)
    print('>'*35 + ' SimVP Loaded ' + '<'*35)
    
    # predict images w SimVP & generate image files
    # TODO: use val_image_preds directly in dataloader so we can bypass the generation of Images
    print('>'*35 + ' Predicting Last 11 Frames with SimVP ' + '<'*35)
    val_image_preds, image_save_loc = predictSimVP(device, VPmodel, X_val)
    print('>'*35 + ' Generating Images from Predictions ' + '<'*35)
    generateImages(val_image_preds)
    print('>'*35 + ' Images Generated ' + '<'*35)

if __name__ == "__main__":
    main(build_label_set=True)