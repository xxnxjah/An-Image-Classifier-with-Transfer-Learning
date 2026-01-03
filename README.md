# Image Classifier using Transfer Learning

## Project Overview

This project demonstrates how to build an **image classification model using Transfer Learning**. A pre-trained convolutional neural network (CNN) is fine-tuned on a custom dataset to classify images into predefined categories efficiently, even with limited data.

The goal of this project is to showcase practical **Computer Vision**, **deep learning**, and **model deployment–ready inference** skills.

## Key Features

* Uses **Transfer Learning** with a pre-trained CNN (e.g., ResNet18)
* Efficient training on a small dataset
* Robust validation and inference pipeline
* Cross-platform image downloading and preprocessing
* Clean, modular, and reproducible code

## Model Architecture

* **Base Model**: ResNet18 (pre-trained on ImageNet)
* **Custom Head**:

  * Fully connected layer adjusted for the number of target classes
* **Loss Function**: CrossEntropyLoss
* **Optimizer**: SGD

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

## Dataset

The dataset consists of labeled images belonging to the following classes:

* `stop`
* `not_stop`

## Data Preprocessing

Applied transformations include:

* Resize to `224 × 224`
* Conversion to tensor
* Normalization using ImageNet mean and standard deviation

## Model Training

Steps followed during training:

1. Load a pre-trained CNN
2. Freeze base layers
3. Replace the final fully connected layer
4. Train on the custom dataset
5. Monitor training and validation accuracy

## Model Evaluation

* Accuracy computed on validation data
* Visual inspection of predictions
* Model set to `eval()` mode during inference

## Inference on New Images

The trained model can be used to predict unseen images by:

1. Loading the saved model weights
2. Applying the same preprocessing steps
3. Passing the image through the model
4. Mapping output logits to class labels

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

## Results

* High accuracy achieved using transfer learning
* Reduced training time compared to training from scratch
* Stable validation performance with minimal overfitting

## Contributing

Contributions, issues, and feature requests are welcome.

## Contact

**Author**: Najah Ilham
**Role**: Aspiring Computer Vision / Machine Learning Engineer
**LinkedIn**: *linkedin.com/in/najah-ilham*


