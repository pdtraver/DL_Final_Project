from ourConfig import custom_training_config, custom_model_config, custom_training_config_mask, custom_model_config_mask
from openstl.api import BaseExperiment
from openstl.utils import create_parser, default_parser
from dataset.dataloaders import getDataloaders
from openstl.methods import method_maps
import pickle
import numpy as np
import torch
import os
from tqdm import tqdm
from dataset.maskTransformations import convert_to_multi_hot, convert_from_multi_hot
from dataset.dataloaders import CustomDataset
import torchmetrics
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
print('>'*35 + ' loading data ' + '<'*35)

with open('/scratch/pdt9929/DL_Final_Project/dataset/val.pkl', 'rb') as f:
    val = pickle.load(f)
_, _, _, Y_val_mask = val['X_val'], val['Y_val'], val['X_val_mask'], val['Y_val_mask']

print('Ground truth shape: ' + str(np.shape(Y_val_mask)))
Y_val_mask_last_frame = Y_val_mask[:, 10]
print('Ground truth last frames shape: ' + str(np.shape(Y_val_mask_last_frame)))

if os.path.isfile('/scratch/pdt9929/DL_Final_Project/dataset/transformed_val_mask_preds.npy'):
    transformed_preds = np.load('/scratch/pdt9929/DL_Final_Project/dataset/transformed_val_mask_preds.npy')
    print('Transformed prediction shape: ' + str(transformed_preds.shape))
else:
    val_mask_preds = np.expand_dims(np.load('/scratch/pdt9929/TrainUnet/lastval.npy'), axis=2)
    print('Prediction shape: ' + str(np.shape(val_mask_preds)))

    print('>'*35 + ' converting to three hot vectors ' + '<'*35)
    transformed_preds = convert_to_multi_hot(val_mask_preds)
    print('Transformed prediction shape: ' + str(transformed_preds.shape))
    
    with open('/scratch/pdt9929/DL_Final_Project/dataset/transformed_val_mask_preds.npy', 'wb') as f:
        np.save(f, transformed_preds)

transformed_preds = torch.tensor(transformed_preds).float()

# ### SIMVP INFERENCE
# args = create_parser().parse_args([])
# config = args.__dict__
# save_dir = 'dataset/generated'

# # update the training config
# config.update(custom_training_config_mask)
# # update the model config
# config.update(custom_model_config_mask)
# # fulfill with default values
# default_values = default_parser()
# for attribute in default_values.keys():
#     if config[attribute] is None:
#         config[attribute] = default_values[attribute]

# model = method_maps['simvp'](save_dir=save_dir, **config)
# ckpt = torch.load('/scratch/pdt9929/DL_Final_Project/work_dirs/mask_exp/checkpoints/best-epoch=29-val_loss=0.008.ckpt')
# model.load_state_dict(ckpt['state_dict'])
# model.eval()
# model.to(device)

# predictions = torch.zeros((1, 11, 13, 160, 240))
# batch_size = 16

# last = 0
# print('>'*35 + ' Predicting frames ' + '<'*35)
# for i in tqdm(range(int(np.ceil(np.ceil(transformed_preds.shape[0]/batch_size))/2))):
#     if (i+1)*batch_size >= transformed_preds.shape[0]:
#         batch = transformed_preds[i*batch_size:]
#     else:
#         batch = transformed_preds[i*batch_size:(i+1)*batch_size]
#     video_tensors = batch.to(device)
#     with torch.no_grad():
#         pred_frames = model(video_tensors)
    
#     predictions = torch.cat((predictions, pred_frames.to('cpu')), 0)
#     del pred_frames
#     del video_tensors
#     torch.cuda.empty_cache()
#     last = i

# unet_simvp_val_predictions = predictions[1:].cpu().detach()
# with open('dataset/unet_simvp_val_predictions.npy', 'wb') as f:
#     np.save(f, unet_simvp_val_predictions)    

# predictions2 = torch.zeros((1, 11, 13, 160, 240))
# print('>'*35 + ' Predicting frames (batch 2) ' + '<'*35)
# for i in tqdm(range(int(np.floor(np.ceil(transformed_preds.shape[0]/batch_size))/2))):
#     if (i+1+last)*batch_size >= transformed_preds.shape[0]:
#         batch = transformed_preds[(i+last)*batch_size:]
#     else:
#         batch = transformed_preds[(i+last)*batch_size:(i+1+last)*batch_size]
#     video_tensors = batch.to(device)
#     with torch.no_grad():
#         pred_frames = model(video_tensors)
    
#     predictions2 = torch.cat((predictions2, pred_frames.to('cpu')), 0)
#     del pred_frames
#     del video_tensors
#     torch.cuda.empty_cache()

# print('>'*35 + ' Frames predicted ' + '<'*35)
# unet_simvp_val_predictions2 = predictions2[1:].cpu().detach()
# with open('dataset/unet_simvp_val_predictions2.npy', 'wb') as f:
#     np.save(f, unet_simvp_val_predictions2)

print('>'*35 + ' Computing accuracy ' + '<'*35)
def computeJaccard(predictions, ground_truth):
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
    return jaccard(predictions, ground_truth)

unet_simvp_val_predictions = torch.tensor(np.load('dataset/unet_simvp_val_predictions.npy')[:, 10]).unsqueeze(1).to(device)
unet_simvp_val_predictions2 = torch.tensor(np.load('dataset/unet_simvp_val_predictions2.npy')[:, 10]).unsqueeze(1).to(device)

final_preds1 = convert_from_multi_hot(unet_simvp_val_predictions)
final_preds2 = convert_from_multi_hot(unet_simvp_val_predictions2)

final_predictions = np.concatenate((final_preds1, final_preds1))
final_predictions = torch.tensor(final_predictions).squeeze(2).squeeze(1)
ground_truth = torch.tensor(Y_val_mask_last_frame).squeeze(1)
print(final_predictions.shape)
print(ground_truth.shape)
# remove video 01370 -- corrupted
# 1000 --> 0; 1370 --> 370
final_predictions = torch.cat((final_predictions[:370], final_predictions[371:]))
print(computeJaccard(final_predictions, ground_truth))