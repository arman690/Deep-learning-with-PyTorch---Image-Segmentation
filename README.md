# Deep-learning-with-PyTorch---Image-Segmentation
# Human Segmentation Model

This repository contains code for training a human segmentation model using PyTorch and segmentation models from [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch). The model is based on a U-Net architecture with an EfficientNet encoder and is trained to segment humans in images.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset
The dataset used for training consists of images and their corresponding binary masks indicating human segmentation. You can download the dataset [here](https://github.com/parth1620/Human-Segmentation-Dataset-master).

## Installation

To set up the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/human-segmentation-model.git
    cd human-segmentation-model
    ```

2. Install required dependencies:
    ```bash
    pip install torch torchvision
    pip install segmentation-models-pytorch
    pip install albumentations==0.4.6 --upgrade
    pip install opencv-contrib-python
    ```

3. Download the dataset:
    ```bash
    !git clone https://github.com/parth1620/Human-Segmentation-Dataset-master.git
    ```

## Usage

1. **Data Preparation**: The dataset CSV file is loaded, and images are split into training and validation sets. Augmentations are applied to improve generalization.

2. **Training**: Run the training function with the specified model and hyperparameters. Training and validation loss are printed after each epoch.

3. **Evaluation**: After training, the best model is saved and can be evaluated on the validation set to see segmentation results.

4. **Visualization**: Run the visualization code in the notebook to display an original image, the ground truth mask, and the predicted mask side by side.

## Model Training

The model can be trained with the following code:
```python
# Training loop for multiple epochs
best_valid_loss = np.Inf
for epoch in range(EPOCHS):
    train_loss = train_fn(trainloader, model, optimizer)
    valid_loss = eval_fn(validloader, model)

    if valid_loss < best_valid_loss:
        torch.save(model.state_dict(), 'best_model.pth')
        print('Model saved!')
        best_valid_loss = valid_loss

    print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {valid_loss}')
```
Evaluation
```python
To evaluate the model and visualize results:
# Load the best model
model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))

# Choose an image from the validation set
idx = 12
image, mask = validset[idx]

# Predict the segmentation mask
logits_mask = model(image.to(DEVICE).unsqueeze(0))
pred_mask = torch.sigmoid(logits_mask) > 0.5

# Visualize
show_image(image, mask, pred_mask)
```

Results
The model provides a binary segmentation mask for humans in images. Use the evaluation script to visualize and compare the original images, ground truth masks, and predicted masks.
