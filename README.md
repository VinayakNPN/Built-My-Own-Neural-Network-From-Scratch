# 🧠 Neural Network from Scratch

<div align="center">
  
![Neural Network](https://img.shields.io/badge/Neural-Network-blue?style=for-the-badge&logo=brain&logoColor=white)
![Python](https://img.shields.io/badge/python-3.7+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![MIT License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

<img src="assets\image.png" alt="Header Image" width="600"/>


[![Visitor Count](https://visitor-badge.laobi.icu/badge?page_id=vinayak.neural-network-scratch)](https://github.com/vinayak/neural-network-scratch)

*A simple, educational implementation of a neural network built from scratch using only NumPy*

</div>

---

## 🌟 Overview

This repository contains a **pure Python implementation** of a neural network built from scratch without any machine learning frameworks. It's designed to be educational and help you understand the fundamental concepts behind neural networks, including forward propagation, backpropagation, and gradient descent.

### 🎯 What You'll Learn
- How neural networks actually work under the hood
- Forward propagation mechanics
- Backpropagation algorithm implementation
- Gradient descent optimization
- Activation functions (Sigmoid)
- Loss functions (Mean Squared Error)

---

## 🧬 What is a Neural Network?

A **Neural Network** is a computational model inspired by biological neural networks in animal brains. It consists of interconnected nodes (neurons) that process information by passing signals through weighted connections.

### Key Components:
- **Input Layer**: Receives data
- **Hidden Layer(s)**: Process information
- **Output Layer**: Produces results
- **Weights & Biases**: Parameters that the network learns

<div align="center">

```
Input Layer    Hidden Layer    Output Layer
    x₁ ────────── h₁ ─────────── o₁
      \        /  \           /
       \      /    \         /
        \    /      \       /
    x₂ ────────── h₂ ───────
```

</div>

---

## ⚡ What is a Neuron?

A **neuron** (or node) is the basic building block of a neural network. It:

1. **Receives inputs** from previous layer neurons
2. **Applies weights** to each input
3. **Sums** all weighted inputs plus a bias term
4. **Applies an activation function** to produce output

### Mathematical Representation:
```
output = activation(Σ(weight₍ᵢ₎ × input₍ᵢ₎) + bias)
```

### Visual Representation:
```
inputs → [w₁x₁ + w₂x₂ + ... + wₙxₙ + b] → activation → output
```

---

## 📊 The Sigmoid Function

The **Sigmoid function** is an activation function that maps any real number to a value between 0 and 1, making it perfect for binary classification problems.

### Formula:
```
σ(x) = 1 / (1 + e^(-x))
```

### Properties:
- **Range**: (0, 1)
- **Smooth**: Differentiable everywhere
- **S-shaped curve**: Gradual transition from 0 to 1
- **Derivative**: σ'(x) = σ(x)(1 - σ(x))

### Visual Graph:
```
  1.0 |     ╭─────
      |   ╭─╯
  0.5 | ╭─╯
      |╭╯
  0.0 ╯─────────── x
     -5  0   5
```

---

## 🏗️ Network Architecture

Our implementation features a simple yet effective architecture:

```
Architecture: 2-2-1 Network

Input Layer:  2 neurons (x₁, x₂)
Hidden Layer: 2 neurons (h₁, h₂) 
Output Layer: 1 neuron  (o₁)

Total Parameters:
- Weights: 6 (w₁, w₂, w₃, w₄, w₅, w₆)
- Biases:  3 (b₁, b₂, b₃)
```

### Connection Details:
- **h₁ = σ(w₁×x₁ + w₂×x₂ + b₁)**
- **h₂ = σ(w₃×x₁ + w₄×x₂ + b₂)**
- **o₁ = σ(w₅×h₁ + w₆×h₂ + b₃)**

---

## 🔄 Forward Propagation

**Forward propagation** is the process of passing input data through the network to generate predictions.

### Steps:
1. **Input → Hidden Layer**:
   - Calculate weighted sum for each hidden neuron
   - Apply sigmoid activation function

2. **Hidden → Output Layer**:
   - Calculate weighted sum for output neuron
   - Apply sigmoid activation function

### Code Implementation:
```python
def feedforward(self, x):
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1
```

---

## 🔄 Backpropagation

**Backpropagation** is the learning algorithm that adjusts weights and biases to minimize prediction errors.

### Process:
1. **Calculate Loss**: Compare prediction with actual value
2. **Compute Gradients**: Calculate partial derivatives using chain rule
3. **Update Parameters**: Adjust weights and biases using gradient descent

### Key Equations:
- **Loss Function**: `L = (y_true - y_pred)²`
- **Gradient**: `∂L/∂w = ∂L/∂y_pred × ∂y_pred/∂w`
- **Weight Update**: `w_new = w_old - learning_rate × gradient`

---

## 📈 Training Process

### Dataset:
Our network learns to classify based on this sample data:

| Name    | Feature 1 | Feature 2 | Label |
|---------|-----------|-----------|-------|
| Alice   | -2        | -1        | 1     |
| Bob     | 25        | 6         | 0     |
| Charlie | 17        | 4         | 0     |
| Diana   | -15       | -6        | 1     |

### Training Parameters:
- **Learning Rate**: 0.1
- **Epochs**: 1000
- **Loss Function**: Mean Squared Error (MSE)
- **Optimization**: Stochastic Gradient Descent

---

## 🚀 Getting Started

### Prerequisites:
```bash
pip install numpy
```

### Running the Code:
```bash
python neural-net.py
```

### Expected Output:
```
Epoch 0 loss: 0.745
Epoch 10 loss: 0.683
Epoch 20 loss: 0.623
...
Epoch 1000 loss: 0.013
```

---

## 🔍 Code Structure

### Core Functions:

#### **Activation Functions**
```python
def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    """Derivative of sigmoid function"""
    fx = sigmoid(x)
    return fx * (1 - fx)
```

#### **Loss Function**
```python
def mse_loss(y_true, y_pred):
    """Mean Squared Error loss function"""
    return ((y_true - y_pred) ** 2).mean()
```

#### **Neural Network Class**
- `__init__()`: Initialize weights and biases
- `feedforward()`: Forward propagation
- `train()`: Training with backpropagation

---

## 🎓 Interview Questions & Answers

<details>
<summary><strong>🔸 Basic Neural Network Concepts</strong></summary>

**Q1: What is a neural network?**
A: A computational model inspired by biological neural networks, consisting of interconnected nodes that process information through weighted connections.

**Q2: What's the difference between a perceptron and a neural network?**
A: A perceptron is a single-layer neural network, while a neural network typically has multiple layers including hidden layers.

**Q3: Why do we need activation functions?**
A: Activation functions introduce non-linearity, allowing the network to learn complex patterns. Without them, the network would be just a linear regression model.

</details>

<details>
<summary><strong>🔸 Mathematical Foundations</strong></summary>

**Q4: What is the sigmoid function and why is it used?**
A: Sigmoid maps any real number to (0,1), making it suitable for binary classification. Formula: σ(x) = 1/(1 + e^(-x))

**Q5: What is the vanishing gradient problem?**
A: When gradients become extremely small during backpropagation, especially in deep networks with sigmoid activations, making learning very slow.

**Q6: How does backpropagation work?**
A: It calculates gradients of the loss function with respect to weights using the chain rule, then updates weights to minimize loss.

</details>

<details>
<summary><strong>🔸 Training and Optimization</strong></summary>

**Q7: What is the learning rate and how does it affect training?**
A: Learning rate controls how much weights are updated. Too high causes instability, too low causes slow convergence.

**Q8: What's the difference between batch and online learning?**
A: Batch learning updates weights after seeing all training samples, while online learning updates after each sample.

**Q9: How do you prevent overfitting?**
A: Techniques include regularization, dropout, early stopping, and cross