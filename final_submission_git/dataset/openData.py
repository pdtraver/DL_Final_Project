import pickle
import numpy as np

def open_data(directory):
    with open(directory + 'train.pkl', 'rb') as f:
        train = pickle.load(f)

    X_train, Y_train, X_train_mask, Y_train_mask = train['X_train'], train['Y_train'], train['X_train_mask'], train['Y_train_mask']
    print('X_train shape: ' + str(np.shape(X_train)))
    print('Y_train shape: ' + str(np.shape(Y_train)))
    print('X_train_mask shape: ' + str(np.shape(X_train_mask)))
    print('Y_train_mask shape: ' + str(np.shape(Y_train_mask)))
        
    with open(directory + 'val.pkl', 'rb') as f:
        val = pickle.load(f)
        
    X_val, Y_val, X_val_mask, Y_val_mask = val['X_val'], val['Y_val'], val['X_val_mask'], val['Y_val_mask']
    print('X_val shape: ' + str(np.shape(X_val[:499])))
    print('X_test shape: ' + str(np.shape(X_val[499:])))
    print('Y_val shape: ' + str(np.shape(Y_val[:499])))
    print('Y_test shape: ' + str(np.shape(Y_val[499:])))
    print('X_val_mask shape: ' + str(np.shape(X_val_mask[:499])))
    print('X_test_mask shape: ' + str(np.shape(X_val_mask[499:])))
    print('Y_val_mask shape: ' + str(np.shape(Y_val_mask[:499])))
    print('Y_test_mask shape: ' + str(np.shape(Y_val_mask[499:])))
        
    with open(directory + 'hidden.pkl', 'rb') as f:
        X_hidden = pickle.load(f)['X_hidden']
        
    print('Hidden shape: ' + str(np.shape(X_hidden)))
    
    return X_train, Y_train, X_train_mask, Y_train_mask, X_val, Y_val, X_val_mask, Y_val_mask, X_hidden