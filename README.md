# M-Mode Ultrasound Neural Network

A deep learning framework for M-mode ultrasound data processing using transformer and CNN-based autoencoders.

## Overview

This project implements state-of-the-art neural network architectures for analyzing M-mode ultrasound data, focusing on feature extraction and representation learning through autoencoder architectures.

### Key Features

- **Transformer Autoencoder**: Sequence-to-sequence learning with attention mechanisms
- **CNN Autoencoder**: Convolutional approach for spatial feature extraction  
- **Flexible Data Pipeline**: Robust data loading and preprocessing for ultrasound frames
- **Experiment Tracking**: Integration with Weights & Biases for monitoring training
- **Modular Design**: Clean separation of models, training, and data processing

## Installation

### Local Setup

```bash
# Clone the repository
git clone <repository-url>
cd m-mode_nn

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
m-mode_nn/
├── config/           # Configuration files and settings
├── data/            # Data loading and processing modules
├── models/          # Neural network architectures
│   ├── transformerAE.py   # Transformer autoencoder
│   ├── transformerCLS.py  # Transformer classifier
│   └── cnnAE.py          # CNN autoencoder
├── training/        # Training scripts and utilities
├── utils/           # Helper functions and utilities
└── visualization/   # Plotting and visualization tools
```

## Usage

### Training an Autoencoder

```python
python training/train_ae.py --config config/config.yaml
```

### Key Configuration Parameters

- `model_type`: Choose between 'transformer' or 'cnn'
- `batch_size`: Training batch size
- `learning_rate`: Optimizer learning rate
- `num_epochs`: Number of training epochs

## Data Format

Input data follows the format: `[B, T, C, H, W]`
- **B**: Batch size
- **T**: Temporal dimension (sequence length)
- **C**: Channels (3 for ultrasound data)
- **H**: Height (132 A-mode samples)
- **W**: Width (5 A-mode scanlines)

## Models

### TransformerAutoencoder
- Pure attention-based architecture
- Processes sequences of ultrasound windows
- Each window treated as a token

### CNN Autoencoder
- Convolutional layers for spatial feature extraction
- Efficient for local pattern recognition

## License
Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) file for details.

## Acknowledgments
This repository includes software developed by Christoph Leitner at The Integrated Systems Laboratory, ETH Zurich (https://iis.ee.ethz.ch/).

## Contact
Email: christoph.leitner@iis.ee.ethz.ch