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

# Change these hyperparameters to run new experiment
exp_nums = [6]
zero_class_thresholds = [.95]

for idx, exp_num in enumerate(exp_nums):
    zero_class_threshold = zero_class_thresholds[idx]
    print(f'EXPERIMENT NUMBER: {exp_num}')
    print(f'THRESHOLD VALUE: {zero_class_threshold}')
    print(f'EXPORTING TO FOLDER: dataset/val_experiments/exp_{exp_num}/')

    if not os.path.exists(f'dataset/val_experiments/exp_{exp_num}/'):
        os.mkdir(f'dataset/val_experiments/exp_{exp_num}/')
        print(f'Made directory: dataset/val_experiments/exp_{exp_num}/')
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
    print('>'*35 + ' loading data ' + '<'*35)

    with open('/scratch/pdt9929/DL_Final_Project/dataset/val.pkl', 'rb') as f:
        val = pickle.load(f)
    _, _, _, Y_val_mask = val['X_val'], val['Y_val'], val['X_val_mask'], val['Y_val_mask']

    print('Ground truth shape: ' + str(np.shape(Y_val_mask)))
    Y_val_mask_last_frame = Y_val_mask[:, 10]
    print('Ground truth last frames shape: ' + str(np.shape(Y_val_mask_last_frame)))

    if os.path.isfile('/scratch/pdt9929/DL_Final_Project/dataset/transformed_val_mask_preds_w_background.npy'):
        transformed_preds = np.load('/scratch/pdt9929/DL_Final_Project/dataset/transformed_val_mask_preds_w_background.npy')
        print('Transformed prediction shape: ' + str(transformed_preds.shape))
    else:
        val_mask_preds = np.expand_dims(np.load('/scratch/pdt9929/TrainUnet/lastval.npy'), axis=2)
        print('Prediction shape: ' + str(np.shape(val_mask_preds)))

        print('>'*35 + ' converting to three hot vectors ' + '<'*35)
        transformed_preds = convert_to_multi_hot(val_mask_preds)
        print('Transformed prediction shape: ' + str(transformed_preds.shape))
        
        with open('/scratch/pdt9929/DL_Final_Project/dataset/transformed_val_mask_preds_w_background.npy', 'wb') as f:
            np.save(f, transformed_preds)

    transformed_preds = torch.tensor(transformed_preds).float()


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
    ckpt = torch.load('/scratch/pdt9929/DL_Final_Project/work_dirs/mask_exp/checkpoints/best-epoch=44-val_loss=0.009.ckpt')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.to(device)

    macro_batches = [
        transformed_preds[:500],
        transformed_preds[500:]
    ]

    final_preds = np.zeros((1, 1, 1, 160, 240))
    for idx, macro_batch in tqdm(enumerate(macro_batches)):
        if os.path.isfile(f'dataset/exp_{exp_num}/final_val_predictions{idx}_w_background.npy'):
            print("Loading " + f'dataset/exp_{exp_num}/final_val_predictions{idx}_w_background.npy')
            unet_simvp_val_predictions = torch.tensor(np.load(f'dataset/exp_{exp_num}/final_val_predictions{idx}.npy'))
            
        else: 
            if os.path.isfile(f'dataset/unet_simvp_val_predictions{idx}_w_background.npy'):
                print("Loading " + f'dataset/unet_simvp_val_predictions{idx}_w_background.npy')
                unet_simvp_val_predictions = torch.tensor(np.load(f'dataset/unet_simvp_val_predictions{idx}_w_background.npy'))
            
            else:
                macro_batch = torch.tensor(macro_batch).float()
                print('Macro batch shape: ' + str(macro_batch.shape))
                predictions = torch.zeros((1, 11, 14, 160, 240))
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

                unet_simvp_val_predictions = predictions[1:].cpu().detach()
                with open(f'dataset/unet_simvp_val_predictions{idx}_w_background.npy', 'wb') as f:
                    np.save(f, unet_simvp_val_predictions)

            unet_simvp_val_predictions = unet_simvp_val_predictions[:,10].unsqueeze(1)
            print('>'*35 + ' Converting from multi-hot ' + '<'*35)
            print('Macro batch shape: ' + str(unet_simvp_val_predictions.shape))
            print(final_preds.shape)
            
            final_val_predictions = convert_from_multi_hot(unet_simvp_val_predictions.cpu().to(device))
            with open(f'dataset/val_experiments/exp_{exp_num}/final_val_predictions{idx}_w_background.npy', 'wb') as f:
                np.save(f, final_val_predictions)
        
        final_preds = np.concatenate((final_preds, final_val_predictions), axis=0)

    final_val_predictions = final_preds[1:]
    with open(f'dataset/val_experiments/exp_{exp_num}/final_val_predictions_w_background.npy', 'wb') as f:
        np.save(f, final_val_predictions)
        
    final_val_predictions_squeezed = final_val_predictions.squeeze(2).squeeze(1)
    with open(f'dataset/val_experiments/exp_{exp_num}/final_val_predictions_squeezed_w_background.npy', 'wb') as f:
        np.save(f, final_val_predictions_squeezed)
        
        
    print('>'*35 + ' Computing accuracy ' + '<'*35)
    def computeJaccard(predictions, ground_truth):
        jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
        return jaccard(predictions, ground_truth)

    final_predictions = torch.tensor(final_val_predictions_squeezed)
    ground_truth = torch.tensor(Y_val_mask_last_frame).squeeze(1)
    print(final_predictions.shape)
    print(ground_truth.shape)
    # remove video 01370 -- corrupted
    # 1000 --> 0; 1370 --> 370
    final_predictions = torch.cat((final_predictions[:370], final_predictions[371:]))
    print(computeJaccard(final_predictions, ground_truth))