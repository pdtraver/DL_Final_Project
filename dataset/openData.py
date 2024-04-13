import pickle
import numpy as np

with open('dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)
    
print(np.shape(dataset))