import torch
import numpy as np
import torchmetrics
import pickle

predictions = np.load('/scratch/pdt9929/TrainUnet/lasthidden.npy')
print(np.shape(predictions))


with open('/scratch/pdt9929/DL_Final_Project/dataset/val.pkl', 'rb') as f:
    val = pickle.load(f)
_, _, _, Y_val_mask = val['X_val'], val['Y_val'], val['X_val_mask'], val['Y_val_mask']

print('Ground truth shape: ' + str(np.shape(Y_val_mask)))
Y_val_mask_last_frame = Y_val_mask[:, 10]
print('Ground truth last frames shape: ' + str(np.shape(Y_val_mask_last_frame)))

predictions = np.load('/scratch/pdt9929/TrainUnet/val_predictions_unet.npy')

print(np.shape(Y_val_mask_last_frame))
print(np.shape(predictions))

def computeJaccard(predictions, ground_truth):
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
    return jaccard(predictions, ground_truth)

print(computeJaccard(torch.tensor(predictions), torch.tensor(Y_val_mask_last_frame).squeeze(1)))