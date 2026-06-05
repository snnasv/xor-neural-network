# Pure NumPy Neural Network for XOR

**Motivation:** To establish a rigorous mathematical understanding of Multi-Layer Perceptrons (MLPs) by building a neural network entirely from scratch, without relying on high-level deep learning frameworks like PyTorch or TensorFlow.

This project implements a minimalist, two-layer Artificial Neural Network in pure Python and NumPy. It successfully solves the classic **XOR (Exclusive OR) Problem**, demonstrating the network's ability to model non-linear decision boundaries—a task impossible for a single-layer perceptron.

## Core Features
- **Matrix Operations:** Forward propagation implemented strictly using pure matrix dot products.
- **Backpropagation:** Manual computation of gradients via the chain rule to update weights iteratively.
- **Bias Integration:** Implementation of bias terms to overcome origin-bound linear separation limitations.
- **Activation Dynamics:** Application of the Sigmoid function and its analytical derivative for continuous gradient calculation.

## Empirical Results
After 10,000 training epochs, the network converges successfully, predicting the XOR logic gates with high precision:

| Input (X1, X2) | Target | Network Output |
| :---: | :---: | :---: |
| 0, 0 | 0 | ~ 0.01 |
| 0, 1 | 1 | ~ 0.98 |
| 1, 0 | 1 | ~ 0.98 |
| 1, 1 | 0 | ~ 0.01 |

## Usage
Ensure `numpy` is installed in your environment, then execute the script:
```bash
python xor_neural_network.py
