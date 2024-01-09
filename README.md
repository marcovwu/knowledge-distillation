# Knowledge Distillation Loss

This repository contains PyTorch implementation of various Knowledge Distillation (KD) losses for training deep neural networks using knowledge from a teacher model to enhance the performance of a student model. The detail KD losses introduction are in the [slides](https://kneron-my.sharepoint.com/:p:/g/personal/marco_wu_kneron_us/EXpnsuaMiaZDggbKsR0wi7IBj5NdJvLpxUMqOkPGlc3ZGg?e=85M9MD).

## Overview
Knowledge Distillation is a technique that involves transferring knowledge from a large, well-trained model (teacher) to a smaller model (student). This helps the student model generalize better and improve its performance on various tasks.

## Implemented KD Losses

| Loss Type               | Description                                             | Use Case                                          |
|-------------------------|---------------------------------------------------------|---------------------------------------------------|
| DINOLoss (DINO)         | Unsupervised representation learning loss.               | Feature learning without labels.                  |
| Binary Cross Entropy (BCE) | Binary cross-entropy loss for binary classification. | Binary classification tasks.                       |
| Mean Squared Error (MSE) | Mean squared error loss for regression tasks.         | Regression tasks.                                 |
| Smooth L1 Loss (SML)     | Smooth L1 loss, suitable for regression with outliers. | Regression tasks with outliers.                    |
| DKDLoss (DKD)           | Combines Targeted Class Knowledge Distillation (TCKD) and Non-targeted Class Knowledge Distillation (NCKD). | Balanced knowledge transfer for classification tasks. |
| KDLoss (KLD)            | Knowledge Distillation loss using Kullback-Leibler Divergence (KLD) for probability distributions. | Transfer knowledge about probability distributions. |
| DIFFKLD                 | Difference Kullback-Leibler Divergence for capturing differences between probability distributions. | Emphasize distribution differences.               |
| SKLD                    | Symmetric Kullback-Leibler Divergence for symmetric comparison of probability distributions. | Symmetric knowledge transfer.                     |
| EU_MSELoss        | Euclidean-based Mean Squared Error Loss.       | Measures the mean squared error between pairwise Euclidean distances of student and teacher features. |
| DynamicCenterNormLoss (DCN) | Dynamic Center Norm Loss for dynamic normalization in knowledge distillation. | Adaptive normalization for knowledge transfer.   |
| CRCDLoss (CRCD)         | Contrastive Regional Context Distillation Loss for enhancing regional context knowledge distillation. | Capture contextual information for better distillation. |
| NKDLoss (NKD)           | Non-targeted Knowledge Distillation Loss.               | General knowledge transfer without targeting specific classes. |
| USKDLoss (USKD)         | Unsupervised Knowledge Distillation Loss.               | Knowledge transfer without labeled data.         |
| hcl loss (RKD)               | Distilling Knowledge via Knowledge Review.       | Captures knowledge at different spatial regions. |

## Usage

### Prerequisites
Python 3.x
PyTorch
Other dependencies (install via pip install -r requirements.txt)

### Installation
Clone the repository:
```
git clone https://github.com/your_username/knowledge-distillation-loss.git
cd knowledge-distillation-loss
```

### Example Usage

```
import torch
from make_loss import make_loss

# Define your configuration and model parameters
config = {
    'MODEL': {
        'FEAT_DIM': 256,
        'METRIC_LOSS_TYPE': 'kd-dino-bce-dkd',
        'KD_LOSS_WEIGHT': [1.0, 1.0, 1.0],
        # Add other model parameters...
    },
    'SOLVER': {
        'MAX_EPOCHS': 100,
        # Add solver parameters...
    },
}

num_classes = 10  # Specify the number of classes in your classification task
num_data = 1000  # Specify the number of data samples in your dataset

# Create the KD loss function
kd_loss_func = make_loss(config, num_classes, num_data)

# Example usage within your training loop
bs = 128  # batch size example
feat_dim = config['MODEL']['FEAT_DIM']
for epoch in range(config['SOLVER']['MAX_EPOCHS']):
    # Get model predictions, features, and other necessary inputs
    # Replace the following placeholders with actual data
    fea_stu, fea_tea, feat_stu, feat_tea = torch.randn(bs, 512), torch.randn(bs, 512), torch.randn(bs, feat_dim), torch.randn(bs, feat_dim)
    feat_stu, feats_tea = [
        torch.randn(bs, 24, 48, 16), torch.randn(bs, 64, 24, 8), torch.randn(bs, 160, 12, 4)
    ], [
        [torch.randn(bs, 24, 48, 16), torch.randn(bs, 64, 24, 8), torch.randn(bs, 160, 12, 4)]
    ]
    score_stu, score_tea, target = torch.randn(bs, num_classes), torch.randn(bs, num_classes), torch.randint(0, num_classes, (bs,))

    # Compute KD loss
    kd_loss = kd_loss_func(score=score_stu, feat={
        'fea_stu': fea_stu, 'fea_tea': fea_tea, 'stu': feat_stu, 'tea': feat_tea,
        'feats_stu': feas_stu, 'feats_tea': feats_tea, 'score_stu': score_stu, 'score_tea': score_tea
        }, fea=None, target=target, epoch=epoch
    )

    # Further steps for backpropagation and optimization...
```

###  Customization
Feel free to customize the config dictionary and model parameters based on your specific use case. You can adjust the number of classes, feature dimensions, and other settings to match your model architecture and training requirements.

## Contributors
Chia Wei Wu (@marcovwu)

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
Acknowledge any relevant sources, frameworks, or libraries used in your project.
Give credit to the authors of papers or implementations that inspired or influenced your work.