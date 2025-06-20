# DPUL(Federated Deep Unlearning)
## About The Project
DPUL is a framework for federated deep unlearning, which allows for the removal of specific data from machine learning models in a federated learning setting.
## Presented Unlearning method:
- **MP(Memory Process)**: The MP mehtod is to regress the parameter to the state without high-weight contribution.
- **DU(Deep Unlearning)**: The DU method aims to remove the influence of low-weight contribution with assisstance of reconstrucion network. Refactoring all parameters using a reconstruction network.
- **PBR(Projected Boost Recovery)**: The PBR method is to recovery the unlearning model with help of projection method.
## Getting Started
### Requirements
### Requirements
| Package      | Version      |
|--------------|--------------|
| torch        | 1.12.1+cu113 |
| torchvision  | 0.13.1+cu113 |
| python       | 3.10.12      |
| numpy        | 1.23.5       |
| peft         | 0.14.0       |
| tqdm         | 4.67.1       |
| matplotlib   | 3.9.3        |
| transformers | 4.47.0       |
| pandas       | 2.2.3        |
| pillow       | 11.0.0       |

### File Structure
```
├─data
│    └─ datasets.txt
│      
├─models
│    ├─  FedAvg.py
│    ├─  seed.py
│    ├─  test.py
│    ├─  Update.py
│    ├─  VAE.py
│    └─  Vectomodel.py
│ 
│          
├─utils
│    ├─  load_datasets.py
│    ├─  options.py
│    └─  sample.py
│    
├─ DPUL_p1.py
├─ DPUL_p2.py
├─ FL.py
└─ README.md
```
There are severl parts in the code:
- data folder: This folder contains the training and testing data forthe target model. In order to reduce the memory space, we just list thelinks to theset dataset here.

-- CIFAR10 and CIFAR100 download from PyTorch

-- CINIC10 download link: https://datashare.ed.ac.uk/download/DS_10283_3192.zip

-- ImageNetTiny download link: https://cs231n.stanford.edu/tiny-imagenet-200.zip

-- VIT-small download link: https://huggingface.co/WinKawaks/vit-small-patch16-224/tree/main

-- VIT-base download link: https://huggingface.co/google/vit-base-patch16-224-in21k/tree/main

-- Deit-base download link: https://huggingface.co/facebook/deit-base-distilled-patch16-224

-- VIT-large download link: https://huggingface.co/google/vit-large-patch16-224-in21k

- models folder: 
This folder contains the implementation of the VAE model, the federated learning algorithm, and the unlearning algorithm.

-- The federated learning algorithm is implemented in the Update.py file.

-- The test.py file is used to test the performance of the target model.

-- The seed.py file is used to set the random seed for reproducibility.

-- The VAE.py file is used to implement the VAE model for the DU method.

-- The Vectomodel.py file is used to implement the vector model for the MP method.

- utils folder:
utils folder: This folder contains the implementation of the dataset loader, the options parser, and the sample loader.

-- The load_datasets.py file is used to load the datasets.

-- The options.py file is used to parse the options.

-- The sample.py file is used to load the sample.

- DPUL_p1.py: This file is used to implement the DPUL method with the MP method and DU method.

- DPUL_p2.py: This file is used to implement the DPUL method with the PBR method.

- FL.py: This file is used to implement the federated learning algorithm.

## Parameter Setting of DPUL
--epochs: The number of FL training epochs. The default value is 50.

--num_users: The number of users (clients) participating in FL. The default value is 10.

--frac: The fraction of clients selected in each training round. The default value is 1.

--local_ep: The number of local epochs per client. The default value is 1.

--local_bs: The local batch size used during client training. The default value is 128.

--bs: The batch size used for testing. The default value is 128.

--lr: The learning rate for training. The default value is 0.001.

--momentum: The momentum used in SGD optimization. The default value is 0.9.

--split: The type of train-test split (user or sample). The default value is 'user'.

--model: The model architecture to be used (e.g., cnn). The default value is 'cnn'.

--kernel_num: The number of each kind of convolutional kernel. The default value is 9.

--kernel_sizes: Comma-separated kernel sizes for convolution. The default value is '3,4,5'.

--norm: The normalization method used (batch_norm, layer_norm, or None). The default value is 'batch_norm'.

--num_filters: The number of filters in the convolutional layers. The default value is 32.

--max_pool: Whether to use max pooling instead of strided convolutions. The default value is 'True'.

--dataset: The name of the dataset used for training. The default value is 'mnist'.

--iid: Whether the dataset is independent and identically distributed (i.i.d). Default is False when used.

--num_classes: The number of output classes. The default value is 10.

--num_channels: The number of image channels. The default value is 1.

--gpu: The ID of the GPU to use. Set -1 to use CPU. The default value is 0.

--stopping_rounds: The number of rounds for early stopping. The default value is 10.

--verbose: Enable verbose output. Default is True unless the flag is specified.

--seed: The random seed for initialization. The default value is 1.

--all_clients: Whether to aggregate updates from all clients. Default is False unless the flag is specified.

--AE_epochs: The number of VAE training epochs. The default value is 100.

--slices: The number of slices to split parameter into. The default value is 10.

--post_epochs: The number of training epochs in post-processing. The default value is 50.

--beta: The beta parameter, for VAE loss coefficient. The default value is 0.5.

--lambda_: The lambda parameter, for High-weight coefficient. The default value is 6.

## Execute DPUL
Edit FL.py, DPUL_p1.py or DPUL_p2.pyfiles, modify parameters, such as datasets, epochs, model and so on, and run FL first, then run DPUL_p1, finally run DPUL_p2.

