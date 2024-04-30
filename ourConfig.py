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
    'total_length': 22,
    'batch_size': 1,
    'val_batch_size': 1,
    'epoch': 100,
    'lr': 0.001,   
    'metrics': ['mse', 'mae'],
    
    'ex_name': 'mask_exp',
    'dataname': 'masks',
    'in_shape': [11, 13, 160, 240],
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