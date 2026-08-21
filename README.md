# DPUL: Dual-Phase Federated Deep Unlearning via Weight-Aware Rollback and Reconstruction

## 📄 Paper Links

- **arXiv**: https://arxiv.org/abs/2512.13381
- **IEEE Xplore**: https://doi.org/10.1109/INFOCOM59046.2026.11571665
- **Conference**: IEEE INFOCOM 2026

---

## 📝 Abstract

Federated Unlearning (FUL) focuses on client data and computing power to offer a privacy-preserving solution. However, high computational demands, complex incentive mechanisms, and disparities in client-side computing power often lead to long waiting time and high costs. To address these challenges, many existing methods rely on server-side knowledge distillation that solely removes the updates of the target client, overlooking the privacy embedded in the contributions of other clients, which can lead to privacy leakage. In this work, we introduce DPUL, a novel server-side unlearning method that deeply unlearns all influential weights to prevent privacy pitfalls. Our approach comprises three components: (i) identifying high-weight parameters by filtering client update magnitudes, and rolling them back to ensure deep removal, (ii) leveraging the variational autoencoder (VAE) to reconstruct and eliminate low-weight parameters, (iii) utilizing a projection-based technique to recover the model. Experimental results on four datasets demonstrate that DPUL surpasses state-of-the-art baselines, providing a 1%–5% improvement in accuracy and up to 12× reduction in time cost.

**Index Terms**—Federated unlearning, large model, deep unlearning, data trading.

---

## 🎯 Key Results

### Overall Method Comparison

![Fig. 1: DPUL vs Traditional Methods](https://raw.githubusercontent.com/00taotao/DPUL/main/README_figures/fig1.jpeg)

*Traditional federated unlearning requires client participation, while DPUL operates entirely server-side.*

### Method Overview

![Fig. 3: DPUL Full Workflow](https://raw.githubusercontent.com/00taotao/DPUL/main/README_figures/fig3.jpeg)

*Three phases: Memory Rollback → Reconstruction Unlearning (β-VAE) → Projected Boost Recovery*

### Accuracy Recovery (4 Datasets)

| Dataset | DPUL | FA | Improvement |
|---------|------|------|-------------|
| CIFAR-10 | **88.82%** | 87.79% | +1.03% |
| CINIC-10 | **83.21%** | 82.13% | +1.08% |
| CIFAR-100 | **68.12%** | 51.35% | +16.77% |
| ImageNet-tiny | **69.92%** | 35.41% | +34.51% |

![Fig. 4: Accuracy Recovery Performance](https://raw.githubusercontent.com/00taotao/DPUL/main/README_figures/fig4.png)

### Efficiency

- **~12× speedup** vs Retrain
- **~4× speedup** vs FE/RR
- Runtime independent of client count

![Fig. 6: Time Consumption Analysis](https://raw.githubusercontent.com/00taotao/DPUL/main/README_figures/fig6.png)

### Unlearning Effectiveness (Backdoor Attack)

- Attack accuracy reduced to **0.58%–4.18%** (matching Retrain)
- FD shows instability with accuracy increasing in some rounds

![Fig. 7: Backdoor Attack Verification](https://raw.githubusercontent.com/00taotao/DPUL/main/README_figures/fig7.png)

---

## About The Project

DPUL is a framework for federated deep unlearning, which allows for the removal of specific data from machine learning models in a federated learning setting.

### Presented Unlearning Methods:

- **MP (Memory Process)**: Regresses parameters to the state without high-weight contribution.
- **DU (Deep Unlearning)**: Removes the influence of low-weight contributions with assistance of a reconstruction network, refactoring all parameters.
- **PBR (Projected Boost Recovery)**: Recovers the unlearning model with the help of projection method.

---

## Getting Started

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

There are several parts in the code:

- **data folder**: Contains training and testing data links. To reduce memory space, we list download links here.
  - CIFAR-10 and CIFAR-100: download from PyTorch
  - CINIC-10: https://datashare.ed.ac.uk/download/DS_10283_3192.zip
  - ImageNetTiny: https://cs231n.stanford.edu/tiny-imagenet-200.zip
  - ViT-small: https://huggingface.co/WinKawaks/vit-small-patch16-224/tree/main
  - ViT-base: https://huggingface.co/google/vit-base-patch16-224-in21k/tree/main
  - DeiT-base: https://huggingface.co/facebook/deit-base-distilled-patch16-224
  - ViT-large: https://huggingface.co/google/vit-large-patch16-224-in21k

- **models folder**: Contains the implementation of the VAE model, federated learning algorithm, and unlearning algorithm.
  - `Update.py`: Federated learning algorithm
  - `test.py`: Test model performance
  - `seed.py`: Set random seed for reproducibility
  - `VAE.py`: VAE model for the DU method
  - `Vectomodel.py`: Vector model for the MP method

- **utils folder**: Contains dataset loader, options parser, and sample loader.
  - `load_datasets.py`: Load datasets
  - `options.py`: Parse command-line options
  - `sample.py`: Load samples

- `DPUL_p1.py`: Implements DPUL with MP and DU methods
- `DPUL_p2.py`: Implements DPUL with PBR method
- `FL.py`: Federated learning algorithm

---

## Parameter Setting of DPUL

| Parameter | Default | Description |
|-----------|---------|-------------|
| --epochs | 50 | Number of FL training epochs |
| --num_users | 10 | Number of clients |
| --frac | 1 | Fraction of clients per round |
| --local_ep | 1 | Local training epochs |
| --local_bs | 128 | Local batch size |
| --bs | 128 | Testing batch size |
| --lr | 0.001 | Learning rate |
| --momentum | 0.9 | SGD momentum |
| --split | 'user' | Train-test split type |
| --model | 'cnn' | Model architecture |
| --dataset | 'mnist' | Dataset name |
| --iid | False | Whether dataset is i.i.d |
| --gpu | 0 | GPU ID (-1 for CPU) |
| --beta | 0.5 | VAE loss coefficient |
| --lambda_ | 6 | High-weight coefficient |
| --slices | 10 | Number of parameter slices |
| --AE_epochs | 100 | VAE training epochs |
| --post_epochs | 50 | Post-processing epochs |

---

## Execute DPUL

1. Edit `FL.py`, `DPUL_p1.py` or `DPUL_p2.py`, modify parameters (datasets, epochs, model, etc.)
2. Run FL first: `python FL.py`
3. Then run DPUL_p1: `python DPUL_p1.py`
4. Finally run DPUL_p2: `python DPUL_p2.py`
