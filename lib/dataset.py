import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class ClickPurchaseDataset(Dataset):
    def __init__(self, config):
        self.dataset_name = config['dataset_name']
        self.discount = config['discount']
        self.data = pd.read_pickle(os.path.join(config['data_directory'], 'train.df'))
        
        self.purchase_reward = config['purchase_reward']
        self.click_reward = config['click_reward']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # extract a row from the dataset
        sample = self.data.iloc[idx]
        
        # get features
        current_state = torch.tensor(sample['current_state'], dtype=torch.long)
        next_state = torch.tensor(sample['next_state'], dtype=torch.long)
        action = torch.tensor(sample['action'], dtype=torch.long)
        
        current_state_length = torch.tensor(sample['current_state_length'], dtype=torch.long)
        next_state_length = torch.tensor(sample['next_state_length'], dtype=torch.long)
        
        is_done = torch.tensor(sample['is_done'], dtype=torch.bool)
        
        # compute reward from the clicks/purchases
        reward = 0.0
        if sample['is_purchased']:
            reward = self.purchase_reward
        else:
            reward = self.click_reward
            
        reward = torch.tensor(reward, dtype=torch.float32)
        
        # add discount
        discount = torch.tensor(self.discount, dtype=torch.float32)
        
        return {
            'current_state': current_state,
            'next_state': next_state,
            'action': action,
            'reward': reward,
            'discount': discount,
            'current_state_length': current_state_length,
            'next_state_length': next_state_length,
            'is_done': is_done
        }


class RatingsDataset(Dataset):
    def __init__(self, config):
        self.dataset_name = config['dataset_name']
        self.discount = config['discount']
        self.data = pd.read_pickle(os.path.join(config['data_directory'], 'train.df'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # extract a row from the dataset
        sample = self.data.iloc[idx]
        
        # get features
        current_state = torch.tensor(sample['current_state'], dtype=torch.long)
        next_state = torch.tensor(sample['next_state'], dtype=torch.long)
        action = torch.tensor(sample['action'], dtype=torch.long)
        
        current_state_length = torch.tensor(sample['current_state_length'], dtype=torch.long)
        next_state_length = torch.tensor(sample['next_state_length'], dtype=torch.long)
        
        is_done = torch.tensor(sample['is_done'], dtype=torch.bool)
        
        # compute reward from the ratings
        reward = torch.tensor((sample['rating'] - 3 / 2.0), dtype=torch.float32)
        
        # add discount
        discount = torch.tensor(self.discount, dtype=torch.float32)
        
        return {
            'current_state': current_state,
            'next_state': next_state,
            'action': action,
            'reward': reward,
            'discount': discount,
            'current_state_length': current_state_length,
            'next_state_length': next_state_length,
            'is_done': is_done
        }

