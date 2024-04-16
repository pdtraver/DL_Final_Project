import pickle
import numpy as np
from openstl.utils import show_video_line

def open_data(directory):
    with open(directory + 'train.pkl', 'rb') as f:
        train = pickle.load(f)

    X_train, Y_train, X_train_mask, Y_train_mask = train['X_train'], train['Y_train'], train['X_train_mask'], train['Y_train_mask']
    print('X_train shape: ' + str(np.shape(X_train)))
        
    with open(directory + 'val.pkl', 'rb') as f:
        val = pickle.load(f)
        
    X_val, Y_val, X_val_mask, Y_val_mask = val['X_val'], val['Y_val'], val['X_val_mask'], val['Y_val_mask']
    print('X_val shape: ' + str(np.shape(X_val[:499])))
    print('X_test shape: ' + str(np.shape(X_val[499:])))
        
    with open(directory + 'hidden.pkl', 'rb') as f:
        X_hidden = pickle.load(f)['X_hidden']
        
    print('Hidden shape: ' + str(np.shape(X_hidden)))
    
    return X_train, Y_train, X_train_mask.transpose(0,2,1,3,4), Y_train_mask.transpose(0,2,1,3,4), X_val, Y_val, X_val_mask.transpose(0,2,1,3,4), Y_val_mask.transpose(0,2,1,3,4), X_hidden


# show the given frames from an example
# example_idx = 0
# show_video_line(X_train[example_idx], ncols=11, vmax=0.6, cbar=False, out_path=None, format='png', use_rgb=True)