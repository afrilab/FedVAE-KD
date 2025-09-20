# FedVAE-KD: Privacy-Preserving Federated Learning Framework for Wireless Network Intrusion Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![TensorFlow 2.8+](https://img.shields.io/badge/tensorflow-2.8+-orange.svg)](https://www.tensorflow.org/)
[![Paper Status: Submitted to MDPI](https://img.shields.io/badge/paper-submitted%20to%20MDPI-blue)](https://www.mdpi.com/)

## 📝 Overview

FedVAE-KD is a **novel federated learning framework** designed to address critical security and privacy challenges in wireless network environments. This repository contains the complete implementation of our innovative approach that seamlessly integrates:

- **Variational Autoencoders (VAE)** for efficient feature extraction and dimensionality reduction
- **Knowledge Distillation (KD)** for model compression and communication efficiency
- **Federated Learning (FL)** for distributed training without sharing raw data
- **Advanced Security Mechanisms** including encryption and differential privacy

Our framework introduces significant advancements in collaborative intrusion detection while preserving data privacy and minimizing communication overhead. FedVAE-KD enables **real-time network defense** across distributed wireless infrastructures, making it particularly suitable for modern 5G/6G ecosystems and heterogeneous IoT environments.

> 📝 **Research Impact**: This work has been submitted to MDPI for publication and represents a foundation for future-ready, scalable network security solutions.

## 🔧 Key Features

- **Privacy Preservation**: Implements cryptographic techniques and differential privacy to protect sensitive network data
- **Scalability**: Designed for deployment across distributed wireless networks with varying computational capabilities
- **Efficiency**: Reduces communication overhead by up to 60% through knowledge distillation
- **Robustness**: Includes secure aggregation mechanisms to defend against model poisoning attacks
- **Flexibility**: Supports multiple federated learning algorithms (FedAvg, FedProx, FedAdam)
- **Extensibility**: Modular architecture allows for easy integration of new components

## 🏗️ Repository Structure

```
FedVAE-KD/
├── config/                  # Configuration files in YAML format
│   ├── model.yaml          # Model hyperparameters
│   ├── training.yaml       # Training configuration
│   ├── federated.yaml      # Federated learning settings
│   └── security.yaml       # Security parameters
├── data/                    # Dataset documentation and processing guidelines
├── models/                  # Model storage directory
│   ├── pretrained/         # Pre-trained models (to be added)
│   └── checkpoints/        # Training checkpoints (to be added)
├── notebooks/               # Jupyter notebooks for experimentation
│   ├── experiments/         # Complete workflow examples
│   │   └── fedvae_kd_example.ipynb
│   └── visualization/       # Performance analysis and visualization
├── paper/                   # Research paper and supplementary materials
├── scripts/                 # Command-line scripts for execution
│   ├── run_centralized.py   # Centralized training script
│   └── run_federated.py     # Federated training script
├── src/                     # Source code
│   ├── models/              # Neural network architectures
│   │   ├── vae.py           # Variational Autoencoder implementation
│   │   ├── classifier.py    # Teacher and Student model implementations
│   │   └── federated_vae_kd.py  # Main FedVAE-KD system
│   ├── security/            # Cryptographic utilities and privacy mechanisms
│   │   └── cryptographic_utils.py
│   ├── training/            # Training algorithms and federated learning
│   │   ├── knowledge_distillation.py
│   │   └── federated.py
│   └── utils/               # Helper functions and utilities
│       └── data_utils.py
├── tests/                   # Automated test suite
│   ├── unit/                # Unit tests for individual components
│   └── integration/         # Integration tests for system components
├── environment.yml          # Conda environment specification
├── requirements.txt         # Python package dependencies
├── LICENSE                  # MIT License
├── CITATION.cff             # Citation metadata
├── CONTRIBUTING.md          # Contribution guidelines
└── README.md                # This file
```

## 📊 Datasets

This project utilizes several well-known network intrusion detection datasets for evaluation and experimentation. All datasets are publicly available on Kaggle and have been extensively used in academic research for network security applications.

### Network Intrusion Detection Datasets

| Dataset | Description | Kaggle Link |
|---------|-------------|-------------|
| **KDD Cup 1999** | Classic network intrusion detection dataset with 41 features and 5 million connection records | [View on Kaggle](https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data) |
| **NSL-KDD** | Improved version of KDD Cup 1999 dataset with removed redundancies and duplicates | [View on Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd) |
| **UNSW-NB15** | Modern network traffic dataset with 49 features and 2.5 million records | [View on Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15) |
| **CICDDOS2019** | DDoS attack detection dataset with 88 features and comprehensive attack scenarios | [View on Kaggle](https://www.kaggle.com/datasets/peterfriedrichs/cicddos2019) |

### Dataset Access

To use these datasets in your experiments:

1. Visit the respective Kaggle links above
2. Download the datasets to your local machine
3. Place them in the `data/` directory or specify the path in your configuration files
4. Ensure the datasets are in the correct format (typically CSV) for the framework

> 📝 **Note**: Make sure to comply with Kaggle's terms of use when downloading and using these datasets for your research.

### Citation

When using these datasets in your research, please cite them appropriately:

```bibtex
@misc{kddcup1999_kaggle_2025,
  title={KDD Cup 1999 Dataset},
  author={Kaggle},
  year={2025},
  publisher={Kaggle Datasets}
}

@misc{nslkdd_kaggle_2025,
  title={NSL-KDD Dataset},
  author={Kaggle},
  year={2025},
  publisher={Kaggle Datasets}
}

@misc{unsw_nb15_kaggle_2025,
  title={UNSW-NB15 Dataset},
  author={Kaggle},
  year={2025},
  publisher={Kaggle Datasets}
}

@misc{cicddos2019_kaggle_2025,
  title={CICDDOS2019 Dataset},
  author={Kaggle},
  year={2025},
  publisher={Kaggle Datasets}
}
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.8+
- Compatible with both CPU and GPU environments

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/afrilab/FedVAE-KD.git
   cd FedVAE-KD
   ```

2. Set up the environment using either Conda or pip:

   **Using Conda (recommended):**
   ```bash
   conda env create -f environment.yml
   conda activate fedvae-kd
   ```

   **Using pip:**
   ```bash
   python -m venv fedvae-kd-env
   source fedvae-kd-env/bin/activate  # On Windows: fedvae-kd-env\Scripts\activate
   pip install -r requirements.txt
   ```

3. Verify the installation:
   ```bash
   python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
   python -c "import src.models.vae as vae; print('VAE module loaded successfully')"
   ```

## 🧪 Usage Examples

### Running Experiments

#### Centralized Training
```bash
python scripts/run_centralized.py --config config/model.yaml --data path/to/dataset.csv
```

#### Federated Training
```bash
# Using FedAvg algorithm
python scripts/run_federated.py --config config/federated.yaml --data path/to/dataset.csv --algorithm FedAvg

# Using FedProx algorithm
python scripts/run_federated.py --config config/federated.yaml --data path/to/dataset.csv --algorithm FedProx

# Using FedAdam algorithm
python scripts/run_federated.py --config config/federated.yaml --data path/to/dataset.csv --algorithm FedAdam
```

### Using Jupyter Notebooks

For interactive experimentation, explore the notebooks in the `notebooks/experiments/` directory:

```bash
jupyter notebook notebooks/experiments/fedvae_kd_example.ipynb
```

### Configuration

All experiments are configured through YAML files in the `config/` directory:

- `config/model.yaml`: Model architecture and hyperparameters
- `config/federated.yaml`: Federated learning parameters
- `config/security.yaml`: Security mechanisms configuration
- `config/training.yaml`: Training process settings

## 📊 System Architecture

The FedVAE-KD framework follows a three-stage architecture:

1. **Feature Extraction**: Variational Autoencoder encodes network traffic data into compact latent representations
2. **Knowledge Transfer**: Teacher model trained on centralized data transfers knowledge to lightweight student models
3. **Federated Deployment**: Student models deployed across distributed clients for privacy-preserving collaborative learning

### Security Mechanisms

- **Model Encryption**: Fernet symmetric encryption for secure model weight transmission
- **Differential Privacy**: Gaussian noise injection to prevent membership inference attacks
- **Secure Aggregation**: Weighted averaging with privacy-preserving techniques

## 🧪 Testing

Run the test suite to verify the implementation:

```bash
# Run all unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run specific test files
pytest tests/unit/test_vae.py
```

## 📈 Performance & Achievements

Our comprehensive evaluation demonstrates that FedVAE-KD delivers exceptional results:

### Key Achievements

- **🏆 Outstanding Accuracy**: Achieves 99.92% detection accuracy using only 10% of local data per client
- **⚡ Model Compression**: Delivers 17.2× model compression for resource-constrained edge deployment
- **🚀 Inference Speedup**: Provides 1.10× faster inference while maintaining robust performance
- **🛡️ Security & Privacy**: Ensures formal differential privacy guarantees and adversarial attack resistance
- **📊 Communication Efficiency**: Reduces communication overhead by up to 60% through knowledge distillation
- **📱 Scalability**: Designed for heterogeneous mobile and IoT ecosystems with varying computational capabilities

### Impact & Innovation

FedVAE-KD represents a significant advancement in federated learning for wireless network security. By combining variational autoencoders with knowledge distillation, our framework addresses critical challenges in distributed intrusion detection while maintaining data privacy and minimizing resource requirements.

The framework enables collaborative threat intelligence across distributed wireless infrastructures without compromising data ownership, making it ideal for modern 5G/6G network environments where security and privacy are paramount.

## 📄 Citation

If you use this code or find our work helpful in your research, please cite our paper:

```bibtex
@article{fedvae_kd_2025,
  title={A Privacy-Preserving Federated Learning Framework for Scalable Wireless Network Intrusion Detection},
  author={Yagiz, M.A. and Cengiz, Z.S. and Goktas, P.},
  journal={Submitted to MDPI},
  year={2025}
}
```

You can also cite this repository directly using the CITATION.cff file.

## 📚 References

### Dataset Sources

Kaggle, "KDD Cup 1999 Dataset," Kaggle Datasets, 2025.  
Kaggle, "NSL-KDD Dataset," Kaggle Datasets, 2025.  
Kaggle, "UNSW-NB15 Dataset," Kaggle Datasets, 2025.  
Kaggle, "CICDDOS2019 Dataset," Kaggle Datasets, 2025.

## 🤝 Contributing

We welcome contributions to enhance the FedVAE-KD framework! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:

- Reporting issues
- Suggesting enhancements
- Submitting pull requests
- Code style and testing requirements

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub or contact the maintainers directly.