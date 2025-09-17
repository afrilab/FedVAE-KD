# Pre-trained Models

This directory contains pre-trained models for the FedVAE-KD framework.

## Model Organization

Pre-trained models are organized by:
- Dataset used for training
- Model architecture type
- Training configuration

## Available Models

Currently, this directory is empty. Pre-trained models will be added after training completion.

## Model Formats

Models are saved in TensorFlow SavedModel format for compatibility and ease of deployment.

## Usage

To load a pre-trained model:

```python
import tensorflow as tf

# Load a pre-trained model
model = tf.keras.models.load_model('models/pretrained/model_name')
```

## Adding New Models

To add a new pre-trained model:
1. Create a subdirectory with a descriptive name
2. Save the model using `model.save()` method
3. Include a README.md with model details