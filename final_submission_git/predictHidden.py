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

if os.path.exists('/scratch/pdt9929/DL_Final_Project/dataset/master_hidden_preds.npy'):
    master_hidden_preds = np.load('/scratch/pdt9929/DL_Final_Project/dataset/master_hidden_preds.npy')
else: 
    hidden_dir = '/scratch/pdt9929/DL_Final_Project/dataset/hidden/'
    sorted_hidden_dir = sorted(os.listdir(hidden_dir))
    master_hidden_preds = np.zeros((1, 11, 160, 240))
    for folder in tqdm(sorted_hidden_dir):
        mask = np.expand_dims(np.load(os.path.join(hidden_dir, folder, 'mask.npy')),axis=0)
        master_hidden_preds = np.concatenate((master_hidden_preds, mask), axis=0)
        
    master_hidden_preds = master_hidden_preds[1:]
    with open('/scratch/pdt9929/DL_Final_Project/dataset/master_hidden_preds.npy', 'wb') as f:
        np.save(f, master_hidden_preds)
        
print('Master Hidden Preds Shape: ' + str(master_hidden_preds.shape))

print('>'*35 + ' transforming data (or loading) ' + '<'*35)
if os.path.isfile('/scratch/pdt9929/DL_Final_Project/dataset/transformed_hidden_mask_preds.npy'):
    transformed_preds = np.load('/scratch/pdt9929/DL_Final_Project/dataset/transformed_hidden_mask_preds.npy')
else:
    hidden_mask_preds = np.expand_dims(master_hidden_preds, axis=2)

    print('>'*35 + ' converting to three hot vectors ' + '<'*35)
    transformed_preds = convert_to_multi_hot(hidden_mask_preds)
    
    with open('/scratch/pdt9929/DL_Final_Project/dataset/transformed_hidden_mask_preds.npy', 'wb') as f:
        np.save(f, transformed_preds)
        
print('Transformed prediction shape: ' + str(transformed_preds.shape))


### SIMVP INFERENCE
args = create_parser().parse_args([])
config = args.__dict__
save_dir = 'dataset/generated'

# update the training config
config.update(custom_training_config_mask)
# update the model config
config.update(custom_model_config_mask)
# fulfill with default values
default_values = default_parser()
for attribute in default_values.keys():
    if config[attribute] is None:
        config[attribute] = default_values[attribute]

model = method_maps['simvp'](save_dir=save_dir, **config)
ckpt = torch.load('/scratch/pdt9929/SimVP/best_checkpoint.ckpt')
model.load_state_dict(ckpt['state_dict'])
model.eval()
model.to(device)


macro_batches = [
    transformed_preds[:500],
    transformed_preds[500:1000],
    transformed_preds[1000:1500],
    transformed_preds[1500:2000],
    transformed_preds[2000:2500],
    transformed_preds[2500:3000],
    transformed_preds[3000:3500],
    transformed_preds[3500:4000],
    transformed_preds[4000:4500],
    transformed_preds[4500:],
]

final_preds = np.zeros((1, 1, 1, 160, 240))
for idx, macro_batch in tqdm(enumerate(macro_batches)):
    if os.path.isfile(f'dataset/final_converted_predictions{idx}.npy'):
        print("Loading " + f'dataset/final_converted_predictions{idx}.npy')
        unet_simvp_hidden_predictions = torch.tensor(np.load(f'dataset/final_converted_predictions{idx}.npy'))
        
    else: 
        if os.path.isfile(f'dataset/unet_simvp_hidden_predictions{idx}.npy'):
            print("Loading " + f'dataset/unet_simvp_hidden_predictions{idx}.npy')
            unet_simvp_hidden_predictions = torch.tensor(np.load(f'dataset/unet_simvp_hidden_predictions{idx}.npy'))
        
        else:
            macro_batch = torch.tensor(macro_batch).float()
            print('Macro batch shape: ' + str(macro_batch.shape))
            predictions = torch.zeros((1, 11, 13, 160, 240))
            batch_size = 16

            print('>'*35 + ' Predicting frames ' + '<'*35)
            for i in tqdm(range(int(np.ceil(macro_batch.shape[0]/batch_size))), leave=True):
                if (i+1)*batch_size >= macro_batch.shape[0]:
                    batch = macro_batch[i*batch_size:]
                else:
                    batch = macro_batch[i*batch_size:(i+1)*batch_size]
                video_tensors = batch.to(device)
                with torch.no_grad():
                    pred_frames = model(video_tensors)
                
                predictions = torch.cat((predictions, pred_frames.to('cpu')), 0)
                del pred_frames
                del video_tensors
                torch.cuda.empty_cache()

            unet_simvp_hidden_predictions = predictions[1:].cpu().detach()
            with open(f'dataset/unet_simvp_hidden_predictions{idx}.npy', 'wb') as f:
                np.save(f, unet_simvp_hidden_predictions)

        unet_simvp_hidden_predictions = unet_simvp_hidden_predictions[:,10].unsqueeze(1)
        print('>'*35 + ' Converting from multi-hot ' + '<'*35)
        print('Macro batch shape: ' + str(unet_simvp_hidden_predictions.shape))
        print(final_preds.shape)
        
        final_converted_predictions = convert_from_multi_hot(unet_simvp_hidden_predictions.cpu().to(device))
        with open(f'dataset/final_converted_predictions{idx}.npy', 'wb') as f:
                np.save(f, final_converted_predictions)
    
    final_preds = np.concatenate((final_preds, final_converted_predictions), axis=0)

final_hidden_predictions = final_preds[1:]
with open('dataset/final_hidden_predictions.npy', 'wb') as f:
    np.save(f, final_hidden_predictions)
    
final_hidden_predictions_squeezed = final_hidden_predictions.squeeze(2).squeeze(1)
with open('dataset/final_hidden_predictions_squeezed.npy', 'wb') as f:
    np.save(f, final_hidden_predictions_squeezed)
