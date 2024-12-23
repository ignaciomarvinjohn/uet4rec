# UET4Rec: U-net Encapsulated Transformer for  Sequential Recommender

This is the main repository for the paper "UET4Rec: U-net Encapsulated Transformer for  Sequential Recommender".

You can access the paper [here](https://www.sciencedirect.com/science/article/pii/S0957417424016488).

# Abstract
<div align="justify">
&ensp;&ensp;&ensp;&ensp;&ensp;Recommending a tempting sequence of items according to a user’s previous history of purchases and clicks, for instance, in the online shopping portals is challenging. And yet it is a crucial task for all service providers. One of the core components in the recommender systems is a sequential model with which an input sequence is transformed into the predicted items. Among many, deep neural networks, such as RNN, LSTM, and Transformer, have been favored for this purpose. However, improving the performance of these models remains an important task. To address this, we propose a novel sequential model by combining a U-net and Transformer, called U-net Encapsulated Transformer. This hybrid architecture places a transformer model between a convolutional encoder and its decoder, wherein each convolutional layer processes 1D signal, i.e. text. The primary benefit of this structure is that the computational burden is reduced since the embedding size of the input to the Transformer is drastically decreased as the signal has to go through the multi-layer convolutional encoder. This solution leverages recommendation lists for action predictions and uses user feedback as direct rewards for updating the model. In addition, the loss function mechanism is improved by including contrastive and reinforcement learning losses. Evaluation of the proposed model, including extensive ablation study, is carried out on four standard benchmark datasets, such as RC15, RetailRocket, MovieLens-1M, and Amazon-Beauty, demonstrating superior performance compared to state-of-the-art methods.
</div>

# Updates
- 2024/07/14: Paper is published in Expert Systems with Applications.
- 2024/12/13: Code v1 is uploaded. Includes base code for training.
- 2024/12/23: Code v2 is uploaded. Includes model evaluation.

# Setup

## Dependencies
Our work is developed using the Windows 10 operating system. We used a conda environment with the following dependencies:
- Python 3.10 (needed to open pandas pickle file)
- Cuda Toolkit 11.8
- PyTorch 2.5
- Others: numpy, pandas, tqdm

## Dataset
You can download our sample dataset from this [link](https://drive.google.com/drive/folders/1Oc10Y51UlaoT02l77IMheu9v6oQ0R-BZ?usp=sharing). Take note that this dataset is under the [**CC BY-NC-ND 4.0**](https://creativecommons.org/licenses/by-nc-nd/4.0/) license. Any violation is punishable by law.

If you want to create your custom training dataset, you can make a pandas pickle file in this format:
- current_state (list(int)): contains the sequence input for the current state
- next_state (list(int)): contains the sequence input for the next state
- action (int): item selected by the user in the current state to transition to the next state
- current_state_length (int): number of items in the current state
- next_state_length (int): number of items in the next state
- is_done (boolean): True if the next state is the last sequence, False otherwise

You must add a column to define the rewards depending on the dataset type.
- For a dataset that contains clicks and purchases, add the is_purchased (bool) column.
- For a dataset that contains ratings, add a rating (int) column.

The validation and test datasets are similar in format to the training dataset, just without the next_state, next_state_length, and is_done.

All datasets should be under the *dataset* folder and contain the train.df, val.df, and test.df files. The directory structure should be like:
```
dataset
   |-- retailrocket
   |   |-- train.df
   |   |-- val.df
   |   |-- test.df
   ...
   |-- custom_dataset_name
   |   |-- train.df
   |   |-- val.df
   |   |-- test.df
```

Notes:
- Our code derives the sequence length (block_size) based on the length of the current_state. Therefore, it should be consistent throughout (e.g., all lists must have a length of 10 if $`N=10`$).
- The item (token) numbers should be from $`0,\cdots,V-1`$, where $`V`$ is the number of unique items in the dataset.
- We use the number $`V`$ as the padding token, and thus, there are $`V+1`$ tokens in total.
- Our code assumes that the largest token number in the current_state and next_state is the padding number. Still, you can explicitly define the vocab_size (total number of items) in the config.

## Dataset Class

Dataset classes are defined in lib/dataset.py. They should handle the loading of the samples and the computation of the rewards. We already defined two classes used in our paper.
- ClickPurchaseDataset: for datasets that use clicks/purchases like RC15 and RetailRocket
- RatingsDataset: for datasets that use ratings like Movielens and Beauty

You can create your custom dataset class. Just follow the other two classes for reference.

# Training

For convenience, all variables and hyperparameters are defined in the *config* dictionary inside *train.py*. Here are some important notes:
1. We standardized the terminologies to match the Transformer literature. Thus:
   - *vocab_size* corresponds to the total number of items + the padding token
   - *block_size* corresponds to the sequence length
2. The parameter *experiment_number* indicates a specific run given a dataset. All log files and model files are stored in output/*dataset_name*/*experiment_number* folder.
3. The embedding sizes (emb_1, emb_2, emb_3, emb_4) are set manually, and the U-Net layers in the UET model class are defined separately. We did this so you can easily edit the architecture, such as adding/removing a U-Net layer or changing its functions.

Select the dataset by setting *dataset_id*. If you have a custom dataset, add the folder name inside the *dataset* list and set the appropriate dataset_id.

Once you've set the dataset and config parameters, you can run:
```
python train.py
```

# Customization

Our highly modular code can easily be modified to meet your needs. For example, our training loop in the main program uses the *tran_model()* function, which represents one iteration.

The train_model() calls *forward_pass()* and *compute_loss()* before performing backpropagation.

You can do custom feedforward in the forward_pass() and add or remove loss functions in the compute_loss().

We also use dictionaries to handle inputs and outputs easily.

# Notes
If you have concerns or suggestions regarding our GitHub, don't hesitate to message us. We want to improve this as much as possible, so your comments are welcome!

For inquiries, kindly send an email to mjci@sju.ac.kr.

# Other Links
- **Meme Analysis using LLM-based Contextual Information and U-net Encapsulated Transformer** | [paper](https://ieeexplore.ieee.org/document/10589379) | [Github](https://github.com/ignaciomarvinjohn/meme-uet-hmt)
- **U-Net Encapsulated Transformer for Reducing Dimensionality in Training Large Language Models** | [Github](https://github.com/ignaciomarvinjohn/uetlm)
- **VisUET (tentative)** | [Github](https://github.com/ignaciomarvinjohn/visuet)
- **UETspeech (tentative)** | [Github](https://github.com/ignaciomarvinjohn/uetspeech)


# Citation
```
@article{WANG2024124781,
title = {UET4Rec: U-net encapsulated transformer for sequential recommender},
journal = {Expert Systems with Applications},
volume = {255},
pages = {124781},
year = {2024},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2024.124781},
url = {https://www.sciencedirect.com/science/article/pii/S0957417424016488},
author = {Jia Wang and Marvin John Ignacio and Seunghee Yu and Hulin Jin and Yong-Guk Kim},
keywords = {Recommender, Sequential model, U-net, Transformer, Reinforcement learning, Contrastive learning}
}
```
