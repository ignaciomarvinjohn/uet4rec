# UET4Rec: U-net Encapsulated Transformer for  Sequential Recommender

This is the main repository for the paper "UET4Rec: U-net Encapsulated Transformer for  Sequential Recommender".

You can access the paper [here](https://www.sciencedirect.com/science/article/pii/S0957417424016488).

**Code will be provided this December! (Sorry, we're busy finishing another paper!)**

# Abstract
 <div align="justify">
&ensp;Recommending a tempting sequence of items according to a user’s previous history of purchases and clicks, for instance, in the online shopping portals is challenging. And yet it is a crucial task for all service providers. One of the core components in the recommender systems is a sequential model with which an input sequence is transformed into the predicted items. Among many, deep neural networks, such as RNN, LSTM, and Transformer, have been favored for this purpose. However, improving the performance of these models remains an important task. To address this, we propose a novel sequential model by combining a U-net and Transformer, called U-net Encapsulated Transformer. This hybrid architecture places a transformer model between a convolutional encoder and its decoder, wherein each convolutional layer processes 1D signal, i.e. text. The primary benefit of this structure is that the computational burden is reduced since the embedding size of the input to the Transformer is drastically decreased as the signal has to go through the multi-layer convolutional encoder. This solution leverages recommendation lists for action predictions and uses user feedback as direct rewards for updating the model. In addition, the loss function mechanism is improved by including contrastive and reinforcement learning losses. Evaluation of the proposed model, including extensive ablation study, is carried out on four standard benchmark datasets, such as RC15, RetailRocket, MovieLens-1M, and Amazon-Beauty, demonstrating superior performance compared to state-of-the-art methods.
</div>

# Updates
- 2024/07/14: Paper is published in Expert Systems with Applications.

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
