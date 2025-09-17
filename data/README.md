# Data Directory

This directory contains datasets and data processing scripts for the FedVAE-KD framework.

## Supported Datasets

The FedVAE-KD framework supports multiple network intrusion detection datasets:

1. **UNSW-NB15**: Modern network intrusion detection dataset with 49 features covering various attack categories
2. **CIC-DDoS-2019**: Distributed Denial of Service attack dataset with realistic traffic patterns
3. **HYDRAS**: Large-scale network traffic dataset with diverse attack scenarios
4. **HIKARI**: Flow-based network traffic dataset focusing on specific attack vectors
5. **KDD Cup 1999**: Classic intrusion detection dataset with well-established benchmarks

## Dataset Preparation

To use these datasets with the FedVAE-KD framework:

1. Download the desired dataset and place it in this directory
2. Ensure the dataset is in CSV format
3. Identify the label column (typically named 'label' or 'attack_cat')
4. Preprocess the data using the scripts in `src/data_processing/`

## Data Preprocessing

All datasets undergo standardized preprocessing:

1. **Feature Encoding**: Categorical features are label-encoded
2. **Normalization**: Numerical features are scaled to [0,1] range
3. **Label Processing**: Attack labels are binarized (normal vs attack)
4. **Data Splitting**: Data is split into train/test sets with stratification

## Example Usage

```python
# Load and preprocess data
from src.utils.data_utils import load_dataset, preprocess_data

features, labels = load_dataset('data/unsw-nb15.csv', label_column='label')
features, labels = preprocess_data(features, labels)
```

## Data Privacy

All datasets are processed locally on each client in the federated learning setting, ensuring that raw data never leaves the client device.