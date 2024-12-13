import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UET4Rec(torch.nn.Module):

    def __init__(self, config):
        super(UET4Rec, self).__init__()
        
        # assign attributes
        # self.block_size = config['block_size']
        self.vocab_size = config['vocab_size']
        self.device = config['device']
        
        self.uet_blocks = torch.nn.ModuleList()
        self.ma_blocks = torch.nn.ModuleList()
        self.uet_blocks_ln = torch.nn.ModuleList()
        self.ma_blocks_ln = torch.nn.ModuleList()
        
        # initialize token and position embedding layers
        self.token_embedding_table = nn.Embedding(config['vocab_size'], config['emb_1'])
        self.position_embedding_table = nn.Embedding(config['block_size'], config['emb_1'])
        self.embedding_dropout = nn.Dropout(p=config['transformer_dropout'])
        
        # initialize UET module
        for _ in range(config['n_uet']):
            self.uet_blocks_ln.append(nn.LayerNorm(config['emb_1'], eps=1e-8))
            self.uet_blocks.append(UET(config))
            
        # initialize MA module
        for _ in range(config['n_model_aug']):
            self.ma_blocks_ln.append(nn.LayerNorm(config['emb_1'], eps=1e-8))
            self.ma_blocks.append(ModelAugmentationLayer(config))
        
        # initialize model head
        self.ln_f = nn.LayerNorm(config['emb_1'], eps=1e-8)
        
        # value and policy (action) layers
        self.q_value_layer = nn.Linear(config['emb_1'], config['vocab_size'])
        self.predict_layer = nn.Linear(config['emb_1'], config['vocab_size'])
        
    def forward(self, idx, tok_pos):
        B, N = idx.shape
        
        # embed the input sequence
        # tok_emb = B x N x W
        tok_emb = self.token_embedding_table(idx)
        
        # scale the embeddings
        tok_emb *= self.token_embedding_table.embedding_dim ** 0.5
        
        # add positional encodings
        pos_emb = self.position_embedding_table(torch.arange(N, device=self.device))
        x = tok_emb + pos_emb
        
        # apply dropout
        x = self.embedding_dropout(x)
        
        # forward to UET
        for i in range(len(self.uet_blocks)):
            x_ln = self.uet_blocks_ln[i](x)
            x_uet = self.uet_blocks[i](x_ln)
            x = x_ln + x_uet
            
        # forward to MA
        for i in range(len(self.ma_blocks)):
            x = self.ma_blocks_ln[i](x)
            x = self.ma_blocks[i](x)
            
        # perform the final normalization
        x = self.ln_f(x)
        
        # extract the last valid token from each sequence
        # this represents the "current state"
        batch_indices = torch.arange(len(tok_pos), device=self.device)
        last_valid_indices = tok_pos - 1
        state = x[batch_indices, last_valid_indices, :] # B x W
        
        # project to value and policy (action)
        q_values = self.q_value_layer(state)  # B x vocab_size
        action_selection = self.predict_layer(state)  # B x vocab_size
        
        return {'state': state, 'q_values': q_values, 'action_selection': action_selection}


# UET Architecture (Enc + Trans + Dec)
class UET(nn.Module):
    def __init__(self, config):
        super(UET, self).__init__()
        
        self.transformer_blocks = torch.nn.ModuleList()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels=config['emb_1'], out_channels=config['emb_2'], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(config['emb_2']),
            nn.LeakyReLU(negative_slope=config['negative_slope']),
            nn.Dropout(p=config['unet_dropout'])
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv1d(in_channels=config['emb_2'], out_channels=config['emb_3'], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(config['emb_3']),
            nn.LeakyReLU(negative_slope=config['negative_slope']),
            nn.Dropout(p=config['unet_dropout'])
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv1d(in_channels=config['emb_3'], out_channels=config['emb_4'], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(config['emb_4']),
            nn.LeakyReLU(negative_slope=config['negative_slope']),
            nn.Dropout(p=config['unet_dropout'])
        )
        
        # Transformer
        for _ in range(config['n_layer']):
            self.transformer_blocks.append(CausalTransformerLayer(config['emb_4'], config['n_head'], config['transformer_dropout'], config['pffn_residual']))
            
        # Decoder
        self.dec1 = nn.Sequential(
            nn.Conv1d(in_channels=config['emb_4'], out_channels=config['emb_3'], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(config['emb_3']),
            nn.LeakyReLU(negative_slope=config['negative_slope']),
            nn.Dropout(p=config['unet_dropout'])
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv1d(in_channels=config['emb_3'], out_channels=config['emb_2'], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(config['emb_2']),
            nn.LeakyReLU(negative_slope=config['negative_slope']),
            nn.Dropout(p=config['unet_dropout'])
        )
        
        self.dec3 = nn.Sequential(
            nn.Conv1d(in_channels=config['emb_2'], out_channels=config['emb_1'], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(config['emb_1']),
            nn.LeakyReLU(negative_slope=config['negative_slope']),
            nn.Dropout(p=config['unet_dropout'])
        )
        
    def forward(self, x):
        x = torch.transpose(x, 1, 2) # B x N x W -> B x W x N
        
        # encoder module
        x1 = x
        x = self.enc1(x)
        x2 = x
        x = self.enc2(x)
        x3 = x
        x = self.enc3(x)
        
        # transformer module
        x = torch.transpose(x, 1, 2) # B x W x N -> B x N x W
        
        # get attention mask
        B, N, _ = x.shape
        attn_mask = generate_causal_mask(B, N, x.device)
        
        # perform attention for each transformer block
        for i in range(len(self.transformer_blocks)):
            x = self.transformer_blocks[i](x, attn_mask)
            
        x = torch.transpose(x, 1, 2) # B x N x W -> B x W x N
        
        # decoder module
        x = self.dec1(x) + x3
        x = self.dec2(x) + x2
        x = self.dec3(x) + x1
        
        x = torch.transpose(x, 1, 2) # B x W x N -> B x N x W
        
        return x


# Transformer Module
class CausalTransformerLayer(nn.Module):
    def __init__(self, n_embd, n_head, dropout_rate, pffn_residual):
        super(CausalTransformerLayer, self).__init__()
        
        self.msa = CausalMultiheadAttention(n_embd, n_head, dropout_rate)
        self.pffn = PointWiseFeedForward(n_embd, dropout_rate, pffn_residual)
        self.layer_norm1 = nn.LayerNorm(n_embd, eps=1e-8)
        self.layer_norm2 = nn.LayerNorm(n_embd, eps=1e-8)
        
    def forward(self, x, attn_mask):
        
        # h_1 = MSA(LN(h_0)) + h_0
        x = self.msa(self.layer_norm1(x), attn_mask) + x
        
        # h_2 = PFFN(LN(h_1)) + h_1
        x = self.pffn(self.layer_norm2(x)) + x
        
        return x


def generate_causal_mask(batch_size, block_size, device):
    mask = torch.tril(torch.ones(block_size, block_size))  # lower triangular matrix
    mask = mask.to(device)
    mask = mask.unsqueeze(0) # add a batch dimension
    mask = mask.expand(batch_size, -1, -1)  # expand to match the batch size
    return mask


# MSA Layer
class CausalMultiheadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout_rate):
        super(CausalMultiheadAttention, self).__init__()
        
        self.attention = nn.MultiheadAttention(n_embd, n_head, dropout=dropout_rate)
        
    def forward(self, x, attn_mask):
        # B x N x W -> N x B x W
        x = x.permute(1, 0, 2)
        
        # compute self-attention
        attn_output, _ = self.attention(x, x, x, attn_mask=attn_mask)
        
        # N x B x W -> B x N x W
        attn_output = attn_output.permute(1, 0, 2)
        
        return attn_output


# PFFN Layer
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, n_embd, dropout_rate, pffn_residual):
        super(PointWiseFeedForward, self).__init__()
        
        self.pffn_residual = pffn_residual
        self.conv1 = nn.Conv1d(n_embd, n_embd, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(n_embd, n_embd, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
    def forward(self, x):
        x_res = x
        
        # x = Dropout(Conv(x))
        # Note: B x N x W -> B x W x N since convolution is used
        x = self.dropout1(self.conv1(torch.transpose(x, 1, 2)))
        
        # x = ReLU()
        x = self.relu(x)
        
        # x = Dropout(Conv(x))
        x = self.dropout2(self.conv2(x))
        
        # B x W x N -> B x N x W
        x = torch.transpose(x, 1, 2)
        
        # add residual
        if self.pffn_residual:
            x += x_res
        
        return x


# Model Augmentation Module
class ModelAugmentationLayer(nn.Module):
    def __init__(self, config):
        super(ModelAugmentationLayer, self).__init__()
        
        self.dense = nn.Linear(config['emb_1'], config['emb_1'])
        self.ln = nn.LayerNorm(config['emb_1'], eps=1e-12)
        self.dropout = nn.Dropout(config['ma_dropout'])

    def forward(self, x):
        x_1 = self.dense(x)
        x_1 = F.gelu(x_1)
        x_1 = self.dropout(x_1)
        x_1 += x
        x = self.ln(x_1)

        return x

