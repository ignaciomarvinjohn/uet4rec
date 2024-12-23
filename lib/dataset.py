import os
import pandas as pd
import torch
from torch.utils.data import Dataset


# class for datasets with clicks/purchases data
class ClickPurchaseDataset(Dataset):
    def __init__(self, config, filename='train.df', is_train=True):
        self.dataset_name = config['dataset_name']
        self.data = pd.read_pickle(os.path.join(config['data_directory'], filename))
        self.is_train = is_train
        
        self.discount = config['discount']
        self.purchase_reward = config['purchase_reward']
        self.click_reward = config['click_reward']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return_dict = dict()
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # extract a row from the dataset
        sample = self.data.iloc[idx]
        
        # get common features
        return_dict['current_state'] = torch.tensor(sample['current_state'], dtype=torch.long)
        return_dict['current_state_length'] = torch.tensor(sample['current_state_length'], dtype=torch.long)
        return_dict['action'] = torch.tensor(sample['action'], dtype=torch.long)
        
        # compute reward from the clicks/purchases
        reward = 0.0
        if sample['is_purchased']:
            reward = self.purchase_reward
        else:
            reward = self.click_reward
        return_dict['reward'] = torch.tensor(reward, dtype=torch.float32)
        
        # get additional features in the training set
        if self.is_train:
            return_dict['next_state'] = torch.tensor(sample['next_state'], dtype=torch.long)
            return_dict['next_state_length'] = torch.tensor(sample['next_state_length'], dtype=torch.long)
            return_dict['is_done'] = torch.tensor(sample['is_done'], dtype=torch.bool)
            
            # add discount
            return_dict['discount'] = torch.tensor(self.discount, dtype=torch.float32)
            
            
        return return_dict


# class for datasets with ratings data
class RatingsDataset(Dataset):
    def __init__(self, config, filename='train.df', is_train=True):
        self.dataset_name = config['dataset_name']
        self.data = pd.read_pickle(os.path.join(config['data_directory'], filename))
        self.is_train = is_train
        
        self.discount = config['discount']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return_dict = dict()
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # extract a row from the dataset
        sample = self.data.iloc[idx]
        
        # get common features
        return_dict['current_state'] = torch.tensor(sample['current_state'], dtype=torch.long)
        return_dict['current_state_length'] = torch.tensor(sample['current_state_length'], dtype=torch.long)
        return_dict['action'] = torch.tensor(sample['action'], dtype=torch.long)
        
        # compute reward from the ratings
        return_dict['reward'] = torch.tensor((sample['rating'] - 3 / 2.0), dtype=torch.float32)
        
        # get additional features in the training set
        if self.is_train:
            return_dict['next_state'] = torch.tensor(sample['next_state'], dtype=torch.long)
            return_dict['next_state_length'] = torch.tensor(sample['next_state_length'], dtype=torch.long)
            return_dict['is_done'] = torch.tensor(sample['is_done'], dtype=torch.bool)
            
            # add discount
            return_dict['discount'] = torch.tensor(self.discount, dtype=torch.float32)
            
            
        return return_dict

