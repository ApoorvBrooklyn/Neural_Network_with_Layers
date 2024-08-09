# Neural Network with Layers

This repository contains a simple implementation of a neural network from scratch using only the NumPy library. The model was designed to provide a clear understanding of how a neural network operates internally, without relying on high-level frameworks such as TensorFlow or PyTorch.

## Overview

This project demonstrates the construction of a fully-connected, multi-layer neural network. The primary goal is to implement and train a neural network from scratch, understand the forward and backward propagation processes, and optimize the network using gradient descent.

## Features

- **Custom Neural Network Architecture**: Implemented using only NumPy.
- **Forward and Backward Propagation**: Manually calculated gradients.
- **Gradient Descent Optimization**: Hand-coded to adjust the weights and biases of the network.
- **Configurable Hyperparameters**: Easily modify the number of layers, neurons, learning rate, and activation functions.
- **Training and Evaluation**: Designed to handle a simple dataset for training and validation purposes.

## Repository Structure

- `dataset/` : Contains the dataset used for training the neural network.
- `images/` : Includes visualizations and plots related to the model's performance.
- `.gitignore` : Specifies files and directories to be ignored by Git.
- `Model.py` : The core implementation of the neural network, including all functions for forward pass, backward pass, and optimization.

## Getting Started

### Prerequisites

Ensure you have Python installed along with NumPy. You can install the required packages using:

```bash
pip install numpy 
```

### Running the Model

To train and evaluate the model, simply run the `Model.py` script:

```bash
python Model.py
```

The script will:

- Load the dataset.
- Initialize the neural network with the specified parameters.
- Train the model using gradient descent.
- Evaluate the performance on a validation set.

### Configuration

You can configure the neural network's hyperparameters directly within the `Model.py` script. Parameters such as learning rate, number of layers, and activation functions can be adjusted to experiment with different network setups.

### Results

The trained model achieves an accuracy of over 80% on the provided dataset, demonstrating the effectiveness of a simple neural network even without complex frameworks.

### Visualizations

Refer to the `images/` directory for plots and visualizations of the training process, including loss curves and accuracy graphs.

### Contributing

Contributions are welcome! If you have suggestions for improvements or additional features, feel free to open an issue or submit a pull request.
