# UET4Rec: U-net Encapsulated Transformer for  Sequential Recommender

This is the main repository for the paper "UET4Rec: U-net Encapsulated Transformer for  Sequential Recommender".

You can access the paper [here](https://www.sciencedirect.com/science/article/pii/S0957417424016488).

# Abstract
 <div align="justify">
&ensp;&ensp;&ensp;&ensp;&ensp;Recommending a tempting sequence of items according to a user’s previous history of purchases and clicks, for instance, in the online shopping portals is challenging. And yet it is a crucial task for all service providers. One of the core components in the recommender systems is a sequential model with which an input sequence is transformed into the predicted items. Among many, deep neural networks, such as RNN, LSTM, and Transformer, have been favored for this purpose. However, improving the performance of these models remains an important task. To address this, we propose a novel sequential model by combining a U-net and Transformer, called U-net Encapsulated Transformer. This hybrid architecture places a transformer model between a convolutional encoder and its decoder, wherein each convolutional layer processes 1D signal, i.e. text. The primary benefit of this structure is that the computational burden is reduced since the embedding size of the input to the Transformer is drastically decreased as the signal has to go through the multi-layer convolutional encoder. This solution leverages recommendation lists for action predictions and uses user feedback as direct rewards for updating the model. In addition, the loss function mechanism is improved by including contrastive and reinforcement learning losses. Evaluation of the proposed model, including extensive ablation study, is carried out on four standard benchmark datasets, such as RC15, RetailRocket, MovieLens-1M, and Amazon-Beauty, demonstrating superior performance compared to state-of-the-art methods.
</div>

# Updates
- 2024/07/14: Paper is published in Expert Systems with Applications.
- 2024/12/13: Code v1 is uploaded.

# Setup

## Dependencies
Our work is developed using the Windows 10 operating system. We used a conda environment with the following dependencies:
- Python 3.10 (needed to open pandas pickle file)
- Cuda Toolkit 11.8
- PyTorch 2.5
- Others: numpy, pandas, tqdm

## Dataset
You can download our sample dataset from this [link](https://drive.google.com/drive/folders/1Oc10Y51UlaoT02l77IMheu9v6oQ0R-BZ?usp=sharing). The current code version only supports training. We will upload validation and test datasets once the updated version is uploaded. Take note that this dataset is under the [**CC BY-NC-ND 4.0**](https://creativecommons.org/licenses/by-nc-nd/4.0/) license. Any violation is punishable by law.

If you want to create your custom dataset, you can make a pandas pickle file in this format:
- current_state (list(int)): contains the sequence input for the current state
- next_state (list(int)): contains the sequence input for the next state
- action (int): item selected by the user in the current state to transition to the next state
- current_state_length (int): number of items in the current state
- next_state_length (int): number of items in the next state
- is_done (boolean): True if the next state is the last sequence, False otherwise

You must add a column to define the rewards depending on the dataset type.
- For a dataset that contains clicks and purchases, add is_purchased (bool) column.
- For a dataset that contains ratings, add rating (int) column.

Notes:
- Our code dervies the sequence length (block_size) based on the length of the current_state and next_state. Therefore, it should be consistent throughout (e.g., all lists must have a length of 10 if $`N=10`$).
- The item (token) numbers should be from $`0,\cdots,V-1`$, where $`V`$ is the number of unique items in the dataset.
- We use the number $`V`$ as the padding token, and thus, there are $`V+1`$ tokens in total.

## Dataset Class

Dataset classes are defined in lib/dataset.py. They should handle the loading of the samples and the computation of the rewards. We already defined two classes used in our paper.
- ClickPurchaseDataset: for datasets that use clicks/purchases like RC15 and RetailRocket
- RatingsDataset: for datasets that use ratings like Movielens and Beauty

You can create your custom dataset class. Just follow the other two classes for reference.

# Training

All variables and hyperparameters are defined in the **config** dictionary inside train.py for convenience.

Just run the train.py, and you're ready to go.

# Customization

Our highly modular code can easily be modified to meet your needs. For example, our training loop in the main program uses the **tran_model()** function, which represents one iteration.

The train_model() calls **forward_pass()** and **compute_loss()** before performing backpropagation.

You can do custom feedforward in the forward_pass() and add or remove loss functions in the compute_loss().

We also use dictionaries to handle inputs and outputs easily.

# TODO List

We are still re-coding the evaluation and test. We will upload it immediately once it is done.

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
