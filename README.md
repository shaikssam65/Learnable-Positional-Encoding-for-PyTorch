# Learnable Positional Encoding for Sequence Classification


A PyTorch implementation of learnable positional encoding for transformer-based sequence classification, with comprehensive explanations and dummy dataset support.

## 📝 Table of Contents
- [What is Positional Encoding?](#what-is-positional-encoding)
- [Key Features](#key-features)
- [Installation](#installation)


## 🧠 What is Positional Encoding?
**Positional encoding** injects information about token order into sequence processing models. Unlike RNNs that inherently understand position through sequential processing, transformers require explicit positional information to:
- Preserve sequence order semantics
- Enable distance relationship learning
- Distinguish identical tokens at different positions

**Why Learnable?**  
Traditional approaches use fixed mathematical functions (e.g., sinusoidal). Learnable encodings:
- Adapt to task-specific positional patterns
- Handle variable-length sequences better
- Often outperform fixed encodings in complex tasks

## 🚀 Key Features
- **Flexible Positional Encoding Module**
  - Learnable parameter matrix with normal initialization
  - Automatic sequence length handling
  - Dropout support
- **Complete Training Pipeline**
  - Dummy dataset generator
  - Transformer-based classifier
  - Training/evaluation utilities
- **Extensible Design**
  - Easy integration with existing models
  - Configurable dimensions and hyperparameters

## 💻 Installation
pip install torch
