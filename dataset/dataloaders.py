import torch
from torch.utils.data import Dataset
from dataset.openData import open_data
from dataset.maskTransformations import convert_to_multi_hot, convert_from_multi_hot
import numpy as np
import os

class CustomDataset(Dataset):
    def __init__(self, X, Y, normalize=False, data_name='custom'):
        super(CustomDataset, self).__init__()
        self.X = X
        self.Y = Y
        self.mean = None
        self.std = None
        self.data_name = data_name

        if normalize:
            # get the mean/std values along the channel dimension
            mean = data.mean(axis=(0, 1, 2, 3)).reshape(1, 1, -1, 1, 1)
            std = data.std(axis=(0, 1, 2, 3)).reshape(1, 1, -1, 1, 1)
            data = (data - mean) / std
            self.mean = mean
            self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index]).float()
        labels = torch.tensor(self.Y[index]).float()
        return data, labels

def getDataloaders(directory):
    # openData
    X_train, Y_train, X_train_mask, Y_train_mask, X_val, Y_val, X_val_mask, Y_val_mask, hidden_data = open_data(directory)

    # prepare dataloaders
    batch_size = 1

    train_set = CustomDataset(X=X_train, Y=Y_train)
    print('HERE')
    print(X_train_mask.shape)
    print(Y_train_mask.shape)
    print(X_val_mask.shape)
    print(Y_val_mask.shape)
    print(type(X_train_mask))
    
    # Transform mask data if necessary
    if os.path.isfile('/scratch/pdt9929/DL_Final_Project/dataset/transformed_masks.npz'):
        outfile = np.load('/scratch/pdt9929/DL_Final_Project/dataset/transformed_masks.npz')
        X_train_mask = outfile['X_train_mask']
        Y_train_mask = outfile['Y_train_mask']
        X_val_mask = outfile['X_val_mask']
        Y_val_mask = outfile['Y_val_mask']
        outfile.close()
    else:
        X_train_mask = convert_to_multi_hot(X_train_mask)
        Y_train_mask = convert_to_multi_hot(Y_train_mask)
        X_val_mask = convert_to_multi_hot(X_val_mask)
        Y_val_mask = convert_to_multi_hot(Y_val_mask)
        
        with open('/scratch/pdt9929/DL_Final_Project/dataset/transformed_masks.npz', 'wb') as f:
            np.savez(f, X_train_mask = X_train_mask, Y_train_mask= Y_train_mask, X_val_mask = X_val_mask, Y_val_mask = Y_val_mask)
        
    print('TRANSFORMED')
    print(X_train_mask.shape)
    print(Y_train_mask.shape)
    print(X_val_mask.shape)
    print(Y_val_mask.shape)
    
    train_set_mask = CustomDataset(X=X_train_mask, Y=Y_train_mask)
    val_set = CustomDataset(X=X_val[:499], Y=Y_val[:499])
    val_set_mask = CustomDataset(X=X_val_mask[:499], Y=Y_val_mask[:499])
    test_set = CustomDataset(X=X_val[499:], Y=Y_val[499:])
    test_set_mask = CustomDataset(X=X_val_mask[499:], Y=Y_val_mask[499:])

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    dataloader_train_mask = torch.utils.data.DataLoader(
        train_set_mask, batch_size=batch_size, shuffle=True, pin_memory=True)
    dataloader_val = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    dataloader_val_mask = torch.utils.data.DataLoader(
        val_set_mask, batch_size=batch_size, shuffle=True, pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    dataloader_test_mask = torch.utils.data.DataLoader(
        test_set_mask, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    return dataloader_train, dataloader_train_mask, dataloader_val, dataloader_val_mask, dataloader_test, dataloader_test_mask, hidden_data