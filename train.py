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
    'batch_size': 256, # 120 for Beauty, 256 for the rest
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
    'kernel_size': 5,
    'stride': 1,
    'padding': 2,
    'negative_slope': 0.1,
    'unet_dropout': 0.7, # 0.7 for Beauty, MovieLens, RetailRocket; 0.15 for RC15
    'n_uet': 1, # number of UET modules
    'n_model_aug': 1, # number of MA modules
    'n_head': 1,
    'n_layer': 2, # transformer layer
    'transformer_dropout': 0.1,
    'pffn_residual': True,
    'ma_dropout': 0.5,
    'negative_reward': 1.0,
    'ce_loss_weight': 0.7,
    'dq_loss_weight': 0.7,
    'awac_loss_weight': 1.0,
    'contrastive_loss_weight': 1.0,
    'seed': 0,
    'purchase_reward': 1.0,
    'click_reward': 0.2,
    'topk': [5, 10, 20]
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
        dataloader['train'] = DataLoader(ClickPurchaseDataset(config, 'train.df', is_train=True), batch_size=config['batch_size'], shuffle=True)
        dataloader['val'] = DataLoader(ClickPurchaseDataset(config, 'val.df', is_train=False), batch_size=config['batch_size'], shuffle=False)
        dataloader['test'] = DataLoader(ClickPurchaseDataset(config, 'test.df', is_train=False), batch_size=config['batch_size'], shuffle=False)
        
    elif dataset_id == 2 or dataset_id == 3:
        dataloader['train'] = DataLoader(RatingsDataset(config, 'train.df', is_train=True), batch_size=config['batch_size'], shuffle=True)
        dataloader['val'] = DataLoader(RatingsDataset(config, 'val.df', is_train=False), batch_size=config['batch_size'], shuffle=False)
        dataloader['test'] = DataLoader(RatingsDataset(config, 'test.df', is_train=False), batch_size=config['batch_size'], shuffle=False)
        
    else:
        print("Invalid Dataset ID.")
        quit()
        
    # create a dataloader
    print(f"Done. Dataset type is {dataloader['train'].dataset.__class__.__name__}")
    print(f"Number of training samples: {len(dataloader['train'])}")
    print(f"Number of validation samples: {len(dataloader['val'])}")
    print(f"Number of test samples: {len(dataloader['test'])}")
    
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
        
    # check if the log file exists, if so, remove it
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
        
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
def evaluate_model(main_model, dataloader, config, eval_logger, total_steps, is_final):
    topk = config['topk']
    scores = dict()
    
    # initialize metrics
    scores['total_items'] = 0.0
    scores['total_rewards'] = [0] * len(topk)
    scores['hit_scores'] = [0] * len(topk)
    scores['ndcg_scores'] = [0] * len(topk)
    
    # for each entry in the val/test dataloader
    for batch_idx, batch in enumerate(dataloader):
        
        # get dataset information
        scores['dataset_class'] = dataloader.dataset.__class__.__name__
        if hasattr(dataloader.dataset, 'purchase_reward'):
            scores['purchase_reward'] = getattr(dataloader.dataset, 'purchase_reward')
            
        current_state = batch['current_state'].to(config['device'])
        current_state_length = batch['current_state_length'].to(config['device'])
        
        # predict items and calculate metrics
        with torch.no_grad():
            output = main_model(current_state, current_state_length)
            sorted_list = np.argsort(output['action_selection'].detach().cpu().numpy(), axis=1)
            
        scores = compute_metrics(sorted_list, batch, scores, topk)
        
    # log results
    for i, k in enumerate(topk):
        hr_total = scores['hit_scores'][i] / max(scores['total_items'], 1)
        ndcg_total = scores['ndcg_scores'][i] / max(scores['total_items'], 1)
        
        eval_logger.info(f"{total_steps},{k},{scores['total_rewards'][i]:.4f},{hr_total:.4f},{ndcg_total:.4f}")
        
        # display the final scores
        if is_final:
            print(f"Top-{k} Results:")
            print(f"  - Cumulative Reward: {scores['total_rewards'][i]:.4f}")
            print(f"  - HR: {hr_total:.4f}, NDCG: {ndcg_total:.4f}")
            
    return


# compute all metrics here
def compute_metrics(predictions, batch, scores, topk):
    # get action and rewards
    actions = batch['action'].numpy()
    rewards = batch['reward'].numpy()
    
    # compute Hit Ratio and NDCG
    for i, k in enumerate(topk):
        rec_list = predictions[:, -k:]  # top-k predictions for each sample
        for j in range(len(actions)):  # loop over each sample in the batch
            true_action = actions[j]
            true_reward = rewards[j]
            
            if true_action in rec_list[j]:
                rank = k - np.argwhere(rec_list[j] == true_action).flatten()[0]
                scores['total_rewards'][i] += true_reward
                
                if (scores['dataset_class'] != "ClickPurchaseDataset") or (true_reward == scores['purchase_reward']):
                    scores['hit_scores'][i] += 1.0
                    scores['ndcg_scores'][i] += 1.0 / np.log2(rank + 1)
                    scores['total_items'] += 1.0
    return scores


if __name__ == '__main__':
    #==============================================================
    # Step 0: Setup
    #==============================================================
    main_tic = time.perf_counter()
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
    
    # load weights [optional]
    if config['load_model']:
        print("\n\nLoading the model from checkpoint.")
        checkpoint = torch.load(os.path.join(config['out_directory'], 'model_checkpoint.pth'), weights_only=True)
        
        model['model_1'].load_state_dict(checkpoint['model_1'])
        model['model_2'].load_state_dict(checkpoint['model_2'])
        
        # the old optimizer becomes a warm-up optimizer
        model['m1_warmup_opt'].load_state_dict(checkpoint['opt_1'])
        model['m2_warmup_opt'].load_state_dict(checkpoint['opt_2'])
        
        print("Done.")
        
    #==============================================================
    # Step 3: Model Training
    #==============================================================
    total_steps = 0
    if config['train_model']:
        print("\n\nTraining started.")
        tic = time.perf_counter()
        
        # create output directory and logger
        os.makedirs(config['out_directory'], exist_ok=True)
        train_logger = setup_logger(config['out_directory'] + "train.log", "train_logger")
        train_logger.info("step,ce_loss,dq_loss,awac_loss,contrastive_loss,total_loss")
        eval_logger = setup_logger(config['out_directory'] + "eval.log", "eval_logger")
        eval_logger.info("step,topk,cumulative_reward,hit_rate,ndcg")
        
        # calculate total steps per epoch
        dataset_size = len(dataloader['train'])
        batch_size = config['batch_size']
        steps_per_epoch = (dataset_size + batch_size - 1) // batch_size
        
        # initialize
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
                    log_entry = f"{total_steps},{loss['ce_loss'].item():.6f},{loss['dq_loss'].item():.6f},{loss['awac_loss'].item():.6f},{loss['contrastive_loss'].item():.6f},{loss['total_loss'].item():.6f}"
                    train_logger.info(log_entry)
                    
                # evaluate per 'eval_interval'
                if total_steps % config['eval_interval'] == 0:
                    model['main'].eval()
                    evaluate_model(model['main'], dataloader['val'], config, eval_logger, total_steps, False)
                    model['main'].train()
                    
                # switch optimizer once the switch_interval is reached
                if total_steps == config['switch_interval']:
                    print(f"\nSwitching optimizer learning rate from {config['warmup_lr']} to {config['main_lr']}.")
                    model['m1_optimizer'] = model['m1_main_opt']
                    model['m2_optimizer'] = model['m2_main_opt']
                    
        # store the final train/val values
        log_entry = f"{total_steps},{loss['ce_loss'].item():.6f},{loss['dq_loss'].item():.6f},{loss['awac_loss'].item():.6f},{loss['contrastive_loss'].item():.6f},{loss['total_loss'].item():.6f}"
        train_logger.info(log_entry)
        
        print("\n\nEvaluating the model on validation dataset.")
        model['main'].eval()
        evaluate_model(model['main'], dataloader['val'], config, eval_logger, total_steps, True)
        model['main'].train()
        print("Done.")
        
        toc = time.perf_counter()
        mins = (toc - tic) / 60
        print(f"\nTotal training time: {mins:.4f} minutes")
        
        # save the model
        torch.save({
            'model_1': model['main'].state_dict(),
            'model_2': model['target'].state_dict(),
            'opt_1': model['m1_optimizer'].state_dict(),
            'opt_2': model['m2_optimizer'].state_dict(),
        }, os.path.join(config['out_directory'], 'model_checkpoint.pth'))
        
    #==============================================================
    # Step 4: Model Evaluation
    #==============================================================
    print("\n\nEvaluating the model on test dataset.")
    test_logger = setup_logger(config['out_directory'] + "test.log", "test_logger")
    test_logger.info("step, topk, cumulative_reward, hit_rate, ndcg")
    
    model['main'].eval()
    evaluate_model(model['main'], dataloader['test'], config, test_logger, total_steps, True)
    print("Done.")
    
    main_toc = time.perf_counter()
    mins = (main_toc - main_tic) / 60
    print(f"\nProgram finished after {mins:.4f} minutes.")

