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

The dataset consists of labeled images belonging to the following classes:

* `stop`
* `not_stop`

## Methodology

### Data Preprocessing

- Images were resized to **224 × 224** pixels to match the ResNet-18 input requirements.
- Pixel values were normalized using the ImageNet mean and standard deviation.
- Data augmentation techniques such as random horizontal flipping and random cropping were applied to improve model generalization.

### Model Architecture

#### Why ResNet-18?

ResNet-18 was selected because it is a lightweight residual convolutional neural network that provides strong feature extraction while requiring fewer computational resources than deeper variants such as ResNet-50 or ResNet-101. It is well suited for transfer learning on relatively small datasets.

#### Transfer Learning

Instead of training a convolutional neural network from scratch, this project uses a pretrained ResNet-18 model trained on the ImageNet dataset. The pretrained feature extractor enables the model to learn meaningful visual representations while reducing training time and improving performance.

#### Fine-tuning Strategy

The pretrained convolutional layers were retained, while the final fully connected classification layer was replaced with a new layer corresponding to the number of target classes in the dataset.

(If you froze layers)

During the initial training phase, the pretrained feature extraction layers were frozen while only the final classification layer was trained.

OR

The pretrained network was fine-tuned by updating all layers to better adapt the learned features to the target dataset.

#### Activation Function

The model uses the Rectified Linear Unit (ReLU) activation function throughout the convolutional layers. The final output layer produces class scores that are converted into probabilities using the Softmax function.

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
| Model | ResNet-18 |
| Input Size | 224 × 224 |
| Optimizer | SGD |
| Loss Function | CrossEntropyLoss |
| Classes | 2 |

## Model Evaluation

* Accuracy computed on validation data
* Visual inspection of predictions
* Model set to `eval()` mode during inference

* ## Results

The transfer learning approach successfully classified traffic sign images while requiring significantly less training time than training a convolutional neural network from scratch.

Model evaluation demonstrated stable validation performance with minimal signs of overfitting.

Sample predictions generated during inference are available in the `results/` directory.

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
git clone https://github.com/xxnxjah/resnet-image-classifier.git
cd resnet-image-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open `Image_Classifier.ipynb` in Jupyter Notebook or VS Code and run all cells.

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


