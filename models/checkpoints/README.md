# Training Checkpoints

This directory contains model checkpoints saved during training for the FedVAE-KD framework.

## Checkpoint Organization

Checkpoints are organized by:
- Training date and time
- Model architecture type
- Training configuration

## Checkpoint Contents

Each checkpoint directory typically contains:
- Model weights
- Optimizer state
- Training metadata

## Usage

To restore from a checkpoint:

```python
# In your training script
model.load_weights('models/checkpoints/checkpoint_name/weights.h5')
```

## Automatic Checkpointing

The training scripts automatically save checkpoints based on the configuration in `config/training.yaml`:

```yaml
checkpointing:
  save_frequency: 10  # Save every 10 epochs
  save_best_only: true  # Only save best performing models
  monitor: 'val_accuracy'  # Metric to monitor for best model
```

## Managing Checkpoints

To manage disk space, old checkpoints can be automatically removed by setting:

```yaml
checkpointing:
  max_to_keep: 5  # Keep only the 5 most recent checkpoints
```