import os
import time
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lib.dataset import ClickPurchaseDataset, RatingsDataset
from lib.model import UET4Rec
from lib.loss import dq_loss_fn, awac_loss_fn, infoNCE_loss_fn


dataset = ['rc15','retailrocket','movielens','beauty']
dataset_id = 0


# input configuration
config = {
    'experiment_number': 1,
    'vocab_size': None, # number of unique items
    'load_model': False,
    'train_model': True,
    'num_epochs': 60,
    'batch_size': 32,
    'log_interval': 500,
    'eval_interval': 2000,
    'discount': 0.5, # RL discount factor
    'switch_interval': 30000,
    'warmup_lr': 0.005,
    'main_lr': 0.001,
    'clip_value': 1.0,
    'emb_1': 64,
    'emb_2': 32,
    'emb_3': 16,
    'emb_4': 8,
    'negative_slope': 0.1,
    'unet_dropout': 0.2,
    'n_uet': 1,
    'n_model_aug': 1,
    'n_head': 1,
    'n_layer': 2, # transformer layer
    'transformer_dropout': 0.1,
    'pffn_residual': True, # !!!
    'ma_dropout': 0.5,
    'negative_reward': 1.0,
    'ce_loss_weight': 0.7,
    'dq_loss_weight': 0.7,
    'awac_loss_weight': 1.0,
    'contrastive_loss_weight': 1.0,
    'seed': 0,
    'purchase_reward': 1.0,
    'click_reward': 0.2
}


# define other config variables
def initialize_config(config, dataset):
    # set dataset and directory
    config['dataset_name'] = dataset
    config['data_directory'] = 'dataset/' + dataset + '/'
    config['out_directory'] = 'output/' + dataset + '/' + str(config['experiment_number']) + '/'
    
    # get block size and vocab size in the training dataset
    dataset_df = pd.read_pickle(os.path.join(config['data_directory'], 'train.df'))
    config['block_size'] = dataset_df['current_state'].apply(len).max()
    
    # initialize if not explicitly defined
    if config['vocab_size'] is None:
        config['vocab_size'] = max([max(sublist) for sublist in dataset_df['current_state']] + [max(sublist) for sublist in dataset_df['next_state']]) + 1
        
    return config


# create dataset object and initialize the dataloader
def initialize_dataloader(config, dataset_id):
    dataloader = dict()
    print(f"\n\nLoading {config['dataset_name']} dataset.")
    
    # load the train dataset
    if dataset_id == 0 or dataset_id == 1:
        train_dataset = ClickPurchaseDataset(config)
        
    elif dataset_id == 2 or dataset_id == 3:
        train_dataset = RatingsDataset(config)
        
    else:
        print("Invalid Dataset ID.")
        quit()
        
    # create a dataloader
    print(f"Done. Dataset type is {type(train_dataset)}")
    print(f"Number of samples: {len(train_dataset)}")
    dataloader['train'] = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    return dataloader


# initialize model and optimizers
def initialize_model(config):
    model = dict()
    
    # create models
    print("\n\nCreating UET models.")
    model['model_1'] = UET4Rec(config)
    model['model_2'] = UET4Rec(config)
    
    # create optimizers
    model['m1_warmup_opt'] = torch.optim.Adam(model['model_1'].parameters(), lr=config['warmup_lr'], betas=(0.9, 0.999))
    model['m1_main_opt'] = torch.optim.Adam(model['model_1'].parameters(), lr=config['main_lr'], betas=(0.9, 0.999))
    model['m2_warmup_opt'] = torch.optim.Adam(model['model_2'].parameters(), lr=config['warmup_lr'], betas=(0.9, 0.999))
    model['m2_main_opt'] = torch.optim.Adam(model['model_2'].parameters(), lr=config['main_lr'], betas=(0.9, 0.999))
    
    total_params = sum(p.numel() for p in model['model_1'].parameters())/1e6
    print(f'Done. Total parameters: {total_params:.2f}M')
    
    return model


# setup logger
def setup_logger(log_file_path, name="logger"):
    # configure the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # clear existing handlers if reinitializing
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # file handler for logging to a file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger


# one training step (forward -> loss -> backward)
def train_model(model, batch, config):
    
    # forward pass
    model, forward_output = forward_pass(model, batch, config)
    
    # compute all loss
    loss = compute_loss(forward_output, batch, config)
    
    # backpropagation
    model['optimizer'].zero_grad()
    loss['total_loss'].backward()
    
    # clip gradients
    for param in model['main'].parameters():
        param.grad.data.clamp_(-config['clip_value'], config['clip_value'])
        
    # update model
    model['optimizer'].step()
    
    return model, loss


# perform all feedforward process here
# includes acquiring next q-values, current q-value, action, augmented state, etc.
def forward_pass(model, batch, config):
    output = dict()
    
    # select main and target models
    model = set_main_target_model(model)
    
    # compute for the q-values given the next state
    output['main_q_values'], output['target_q_values'] = compute_q_values(model, batch, config)
    
    # get current state
    current_state = batch['current_state'].to(config['device'])
    current_state_length = batch['current_state_length'].to(config['device'])
    
    # get state, q-value, and action of the main model given the current state
    output['current_main_out'] = model['main'](current_state, current_state_length)
    
    # get augmented state of the main model (for contrastive loss)
    permuted_state = permute_tensor(current_state, axis=1)
    output['augmented_state'] = model['main'](permuted_state, current_state_length)['state']
    
    return model, output


# randomly switch between the two models and assign optimizer
def set_main_target_model(model):
    pointer = np.random.randint(0, 2)
    if pointer == 0:
        model['main'] = model['model_1']
        model['optimizer'] = model['m1_optimizer']
        model['target'] = model['model_2']
    else:
        model['main'] = model['model_2']
        model['optimizer'] = model['m2_optimizer']
        model['target'] = model['model_1']
        
    model['main'].train()   # 'main' is in training mode
    model['target'].eval()  # 'target' is in evaluation mode
    
    return model


# compute for the q-values of main and target using the next state
def compute_q_values(model, batch, config):
    next_state = batch['next_state'].to(config['device'])
    next_state_length = batch['next_state_length'].to(config['device'])
    
    # get main and target q-values based on the next state
    main_out = model['main'](next_state, next_state_length)
    target_out = model['target'](next_state, next_state_length)
    
    # if the episode ends, set target q-values to zero
    is_done = batch['is_done'].tolist()
    for index in range(target_out['q_values'].shape[0]):
        if is_done[index]:
            target_out['q_values'][index] = torch.zeros(config['vocab_size']).to(config['device'])
            
    return main_out['q_values'], target_out['q_values']


# permute the current state
def permute_tensor(tensor, axis=1):
    rng = np.random.default_rng()
    tensor_np = tensor.cpu().numpy()
    permuted_np = rng.permuted(tensor_np, axis=axis)
    return torch.tensor(permuted_np, device=tensor.device)


# compute the total loss
def compute_loss(forward_output, batch, config):
    action = batch['action'].to(config['device'])
    reward = batch['reward'].to(config['device'])
    discount = batch['discount'].to(config['device'])
    loss = dict()
    
    # cross-entropy
    loss['ce_loss'] = F.cross_entropy(forward_output['current_main_out']['action_selection'], action)
    
    # double q-learning
    loss['dq_loss'], qa_state = dq_loss_fn(forward_output, action, reward, discount)
    
    # awac
    loss['awac_loss'] = awac_loss_fn(forward_output, qa_state, action, discount, config['negative_reward'])
    
    # contrastive
    loss['contrastive_loss'] = infoNCE_loss_fn(forward_output['current_main_out']['state'], forward_output['augmented_state'])
    
    # total loss
    loss['total_loss'] = torch.mean(config['ce_loss_weight'] * loss['ce_loss'] +
                                    config['dq_loss_weight'] * loss['dq_loss'] +
                                    config['awac_loss_weight'] * loss['awac_loss'] +
                                    config['contrastive_loss_weight'] * loss['contrastive_loss']
                                   )
    
    return loss


# evaluate model
def evaluate_model(main_model, eval_logger):
    # initialize variables
    
    # set model to eval mode
    main_model.eval()
    
    # load validation dataset
    
    total_steps = 0
    eval_logger.info(f"{total_steps}")
    
    # set model back to train mode
    main_model.train()
    
    return


if __name__ == '__main__':
    #==============================================================
    # Step 0: Setup
    #==============================================================
    np.random.seed(config['seed'])
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = initialize_config(config, dataset[dataset_id])
    
    #==============================================================
    # Step 1: Initialize Dataset/Dataloader
    #==============================================================
    dataloader = initialize_dataloader(config, dataset_id)
    
    #==============================================================
    # Step 2: Initialize Model
    #==============================================================
    model = initialize_model(config)
    model['model_1'].to(config['device'])
    model['model_2'].to(config['device'])
    
    # # load weights [optional]
    # if config['load_model']:
        # print("Loading the model from checkpoint.")
        
    #==============================================================
    # Step 3: Model Training
    #==============================================================
    if config['train_model']:
        print("\n\nTraining started.")
        tic = time.perf_counter()
        
        # create output directory and logger
        os.makedirs(config['out_directory'], exist_ok=True)
        train_logger = setup_logger(config['out_directory'] + "train.log", "train_logger")
        eval_logger = setup_logger(config['out_directory'] + "eval.log", "eval_logger")
        
        # calculate total steps per epoch
        dataset_size = len(dataloader['train'])
        batch_size = config['batch_size']
        steps_per_epoch = (dataset_size + batch_size - 1) // batch_size
        
        # initialize
        total_steps = 0
        model['m1_optimizer'] = model['m1_warmup_opt']
        model['m2_optimizer'] = model['m2_warmup_opt']
        
        # for each epoch
        for epoch in range(config['num_epochs']):
            
            # define progress monitoring for steps and epochs
            epoch_progress = tqdm(dataloader['train'], desc=f"Epoch {epoch + 1}/{config['num_epochs']}")
            
            # for each batch in the dataset
            for batch_idx, batch in enumerate(epoch_progress):
                
                # perform forward and backpropagation
                model, loss = train_model(model, batch, config)
                
                # increment steps for each batch
                total_steps += 1
                
                # log the metrics per 'log_interval'
                if total_steps % config['log_interval'] == 0:
                    log_entry = f"{total_steps}, {loss['ce_loss'].item():.6f}, {loss['dq_loss'].item():.6f}, {loss['awac_loss'].item():.6f}, {loss['contrastive_loss'].item():.6f}, {loss['total_loss'].item():.6f}"
                    train_logger.info(log_entry)
                    
                # # evaluate per 'eval_interval'
                # if total_steps % config['eval_interval'] == 0:
                    # evaluate_model(model['main'], eval_logger)
                    
                # switch optimizer once the switch_interval is reached
                if total_steps == config['switch_interval']:
                    print(f"\nSwitching optimizer learning rate from {config['warmup_lr']} to {config['main_lr']}.")
                    model['m1_optimizer'] = model['m1_main_opt']
                    model['m2_optimizer'] = model['m2_main_opt']
                    
        toc = time.perf_counter()
        mins = (toc - tic) / 60
        print(f"Total training time: {mins:.4f} minutes")
        
        # save the model
        
        
    #==============================================================
    # Step 4: Model Evaluation
    #==============================================================
    print("Evaluating the model.")
    # evaluate_model(model['main'], eval_logger)
    quit()

