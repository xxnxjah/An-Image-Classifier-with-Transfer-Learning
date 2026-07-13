# Image Classifier using Transfer Learning

## Abstract

This project investigates the use of transfer learning for binary image classification using a pretrained ResNet-18 model. By leveraging features learned from ImageNet, the model is fine-tuned on a custom Stop/Not Stop traffic sign dataset, reducing training time while demonstrating the effectiveness of transfer learning for small-scale computer vision tasks.

## Project Overview

This repository contains the implementation of a binary image classification model using ResNet-18 and transfer learning. The project demonstrates an end-to-end deep learning pipeline including dataset preparation, model training, evaluation, and inference.

## Key Features

* Uses **Transfer Learning** with a pre-trained CNN (e.g., ResNet18)
* Efficient training on a small dataset
* Robust validation and inference pipeline
* Cross-platform image downloading and preprocessing
* Clean, modular, and reproducible code

## Dataset

The model was trained on a custom binary traffic sign dataset organized using PyTorch's `ImageFolder` structure.

### Classes

- `stop`
- `not_stop`

### Dataset Split

| Split | Stop | Not Stop | Total |
|-------|-----:|---------:|------:|
| Training | 87 | 90 | 177 |
| Validation | 10 | 10 | 20 |

The dataset was randomly divided into **90% training** and **10% validation** sets while preserving the class-wise directory structure.

## Methodology

### Data Preprocessing

The input images were preprocessed using the following transformations:

- Resized to **224 × 224** pixels to match the input size required by ResNet-18.
- Converted to PyTorch tensors.
- Normalized using the ImageNet mean and standard deviation.

These preprocessing steps ensure compatibility with the pretrained ResNet-18 model while preserving the original image content.

### Model Architecture

#### Why ResNet-18?

ResNet-18 was selected because it is a lightweight residual convolutional neural network that provides strong feature extraction while requiring fewer computational resources than deeper variants such as ResNet-50 or ResNet-101. It is well suited for transfer learning on relatively small datasets.

#### Transfer Learning

Instead of training a convolutional neural network from scratch, this project uses a pretrained ResNet-18 model trained on the ImageNet dataset. The pretrained feature extractor enables the model to learn meaningful visual representations while reducing training time and improving performance.

#### Fine-Tuning Strategy

The pretrained convolutional layers of ResNet-18 were frozen to preserve the features learned from the ImageNet dataset. The original fully connected classification layer was replaced with a new fully connected layer corresponding to the two target classes (`stop` and `not_stop`). During training, only the newly added classifier layer was updated.

#### Activation Function

The ResNet-18 architecture employs the Rectified Linear Unit (ReLU) activation function throughout its convolutional layers. The final classification layer produces class scores that are interpreted using the softmax operation during prediction.

#### Loss Function

CrossEntropyLoss was used as the optimization objective for binary image classification because it combines LogSoftmax and Negative Log Likelihood Loss into a single function.

### Model Training

Steps followed during training:

1. Load a pre-trained CNN
2. Freeze base layers
3. Replace the final fully connected layer
4. Train on the custom dataset
5. Monitor training and validation accuracy

## Experimental Setup

| Parameter | Value |
|------------|--------|
| Framework | PyTorch |
| Base Model | ResNet-18 (ImageNet pretrained) |
| Input Size | 224 × 224 |
| Classes | 2 (`stop`, `not_stop`) |
| Training Images | 177 |
| Validation Images | 20 |
| Epochs | 10 |
| Batch Size | 32 |
| Optimizer | SGD |
| Learning Rate | 0.000001 |
| Momentum | 0.9 |
| Loss Function | CrossEntropyLoss |
| Learning Rate Scheduler | CyclicLR (Triangular2) |
| Device | CPU / CUDA (if available) |

## Model Evaluation

* Accuracy computed on validation data
* Visual inspection of predictions
* Model set to `eval()` mode during inference

## Results

The transfer learning approach successfully learned to distinguish between the `stop` and `not_stop` classes using a relatively small training dataset.

The model was trained for 10 epochs using a frozen pretrained ResNet-18 feature extractor and demonstrated stable validation performance throughout training.

Example predictions generated during inference are available in the `results/` directory.

## Inference on New Images

The trained model can be used to predict unseen images by:

1. Loading the saved model weights
2. Applying the same preprocessing steps
3. Passing the image through the model
4. Mapping output logits to class labels

## Project Structure

```
resnet-image-classifier/
│
├── dataset/                 # Training dataset
├── validation_images/       # Images used for validation/inference
├── results/                 # Saved results and outputs
├── Image_Classifier.ipynb   # Main Jupyter notebook
├── model.pt                 # Trained model weights
├── requirements.txt         # Project dependencies
├── dataset_link.txt         # Dataset source/reference
└── README.md                # Project documentation
```

## Tech Stack

* Python 3.12.0
* PyTorch
* Torchvision
* NumPy
* Pillow (PIL)
* Matplotlib
* Requests
* JupyterLab


## How to Run the Project

### 1.Clone the repository

```bash
git clone https://github.com/xxnxjah/An-Image-Classifier_with-Transfer-Learning.git
cd An-Image-Classifier_with-Transfer-Learning
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open `Transfer_Learning_ResNet18_Image_Classification.ipynb` in Jupyter Notebook or VS Code and run all cells.

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.

2. PyTorch Documentation

3. ImageNet Dataset

## Contributing

Contributions, issues, and feature requests are welcome.

## Contact

**Author**: Najah Ilham
**Role**: Aspiring Computer Vision / Machine Learning Engineer
**LinkedIn**: *linkedin.com/in/najah-ilham*


