# Models Directory

This directory contains all trained models for the FedVAE-KD framework, organized into subdirectories for different purposes.

## Directory Structure

```
models/
├── pretrained/    # Pre-trained models ready for deployment
└── checkpoints/   # Intermediate training checkpoints
```

## Pre-trained Models

The `pretrained/` directory contains fully trained models that are ready for deployment or inference. These models have been validated and tested for performance and accuracy.

## Training Checkpoints

The `checkpoints/` directory contains intermediate snapshots of models during training. These can be used to resume training from a specific point or to analyze training progress.

## Model Formats

All models are saved in TensorFlow's SavedModel format, which provides:
- Language-neutral format
- Recoverable computation graphs
- Compatibility with TensorFlow Serving
- Easy deployment options

## Usage Examples

### Loading a Pre-trained Model

```python
import tensorflow as tf

# Load a pre-trained model
model = tf.keras.models.load_model('models/pretrained/model_name')
```

### Restoring from Checkpoints

```python
# Restore model weights from checkpoint
model.load_weights('models/checkpoints/checkpoint_name/weights.h5')
```

## Adding New Models

When adding new models to this directory:
1. Place pre-trained models in the `pretrained/` subdirectory
2. Place training checkpoints in the `checkpoints/` subdirectory
3. Include appropriate README files with model details
4. Follow the naming conventions for consistency