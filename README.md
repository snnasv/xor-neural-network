# Pure NumPy Neural Network for XOR

A minimalist, two-layer Artificial Neural Network (Multi-Layer Perceptron) built entirely from scratch using only Python and NumPy. No deep learning frameworks (like PyTorch or TensorFlow) were used.

## The Goal
The primary objective of this project is to demonstrate the fundamental mathematics behind deep learning. It successfully solves the classic **XOR (Exclusive OR) Problem**, which is a non-linear problem that a single-layer perceptron cannot solve.

## Features
- **Forward Propagation**: Implemented using pure matrix dot products.
- **Backpropagation**: Calculates gradients manually to update weights.
- **Bias Nodes**: Overcomes the origin-bound linear separation problem to solve XOR.
- **Activation Function**: Sigmoid and its derivative for gradient calculation.

## Results
After 10,000 epochs of training, the network accurately predicts the XOR logic gates:

| Input (X1, X2) | Target | Network Output |
| :---: | :---: | :---: |
| 0, 0 | 0 | ~ 0.01 |
| 0, 1 | 1 | ~ 0.98 |
| 1, 0 | 1 | ~ 0.98 |
| 1, 1 | 0 | ~ 0.01 |

## How to Run
Make sure you have NumPy installed, then simply run the Python file:

```bash
python xor_neural_network.py