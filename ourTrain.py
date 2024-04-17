from ourConfig import custom_training_config, custom_model_config, custom_training_config_mask, custom_model_config_mask
from openstl.api import BaseExperiment
from openstl.utils import create_parser, default_parser
from dataset.dataloaders import getDataloaders
        
# loading data
print('>'*35 + ' loading data ' + '<'*35)
dataloader_train, dataloader_train_mask, dataloader_val, dataloader_val_mask, dataloader_test, dataloader_test_mask, hidden_data = getDataloaders('dataset/') 
        
# design exp, train & test
def run_exp(train, val, test, train_config, model_config):
    args = create_parser().parse_args([])
    config = args.__dict__

    # update the training config
    config.update(train_config)
    # update the model config
    config.update(model_config)
    # fulfill with default values
    default_values = default_parser()
    for attribute in default_values.keys():
        if config[attribute] is None:
            config[attribute] = default_values[attribute]
            
    print('>'*35 + ' designing exp ' + '<'*35)
    exp = BaseExperiment(args, dataloaders=(train, val, test), strategy='auto')

    print('>'*35 + ' training images ' + '<'*35)
    exp.train()

    print('>'*35 + ' testing images  ' + '<'*35)
    exp.test()
    

# run for images & masks
run_exp(dataloader_train, dataloader_val, dataloader_test, custom_training_config, custom_model_config)
#run_exp(dataloader_train_mask, dataloader_val_mask, dataloader_test_mask, custom_training_config_mask, custom_model_config_mask)