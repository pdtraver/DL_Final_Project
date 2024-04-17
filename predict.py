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
from dataset.generateImages import generateImages
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
# loading data
print('>'*35 + ' loading data ' + '<'*35)
with open('dataset/hidden.pkl', 'rb') as f:
        hidden_data = torch.tensor(pickle.load(f)['X_hidden'])
print('Hidden data shape: ' + str(np.shape(hidden_data)))

args = create_parser().parse_args([])
config = args.__dict__
save_dir = 'dataset/generated'


# update the training config
config.update(custom_training_config)
# update the model config
config.update(custom_model_config)
# fulfill with default values
default_values = default_parser()
for attribute in default_values.keys():
    if config[attribute] is None:
        config[attribute] = default_values[attribute]

model = method_maps['simvp'](save_dir=save_dir, **config)
model.load_state_dict(torch.load('work_dirs/image_exp/checkpoints/best-epoch=05-val_loss=0.002.ckpt', map_location=device)['state_dict'])
model.eval()
model.to(device)

predictions = torch.zeros((1, 11, 3, 160, 240))
batch_size = 32

last = 0
for i in tqdm(range(int(np.ceil(np.ceil(hidden_data.shape[0]/batch_size))/2))):
    if (i+1)*batch_size >= hidden_data.shape[0]:
        batch = hidden_data[i*batch_size:]
    else:
        batch = hidden_data[i*batch_size:(i+1)*batch_size]
    video_tensors = batch.to(device)
    with torch.no_grad():
        pred_frames = model(video_tensors)
    
    predictions = torch.cat((predictions, pred_frames.to('cpu')), 0)
    del pred_frames
    del video_tensors
    torch.cuda.empty_cache()
    last = i
    
hidden_predictions = predictions[1:].cpu().detach().numpy()
with open('dataset/hidden_predictions.npy', 'wb') as f:
    np.save(f, hidden_predictions)
    
predictions2 = torch.zeros((1, 11, 3, 160, 240))
torch.cuda.empty_cache()
for i in tqdm(range(int(np.floor(np.ceil(hidden_data.shape[0]/batch_size))/2))):
    if (i+1+last)*batch_size >= hidden_data.shape[0]:
        batch = hidden_data[(i+last)*batch_size:]
    else:
        batch = hidden_data[(i+last)*batch_size:(i+last+1)*batch_size]
    video_tensors = batch.to(device)
    with torch.no_grad():
        pred_frames = model(video_tensors)
    
    predictions2 = torch.cat((predictions, pred_frames.to('cpu')), 0)
    del pred_frames
    del video_tensors
    torch.cuda.empty_cache()
    
hidden_predictions2 = predictions[1:].cpu().detach().numpy()
with open('dataset/hidden_predictions2.npy', 'wb') as f:
    np.save(f, hidden_predictions2)
    
# Create images from predictions
generateImages(hidden_predictions, hidden_predictions2)