import unittest
import numpy as np
import tensorflow as tf
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.federated_vae_kd import FederatedVAEKnowledgeDistillation


class TestFedVAEKDIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a simple config for testing
        self.config = {
            'vae': {
                'input_dim': 10,
                'latent_dim': 5,
                'hidden_layers': [20, 15],
                'dropout_rate': 0.1
            },
            'teacher': {
                'input_dim': 5,
                'hidden_layers': [10],
                'output_dim': 2,
                'dropout_rate': 0.2
            },
            'student': {
                'input_dim': 5,
                'hidden_layers': [8],
                'output_dim': 2,
                'dropout_rate': 0.1
            },
            'training': {
                'epochs': 2,
                'batch_size': 32,
                'optimizer': {
                    'learning_rate': 0.001
                }
            },
            'federated': {
                'num_clients': 3,
                'num_rounds': 2,
                'fraction_clients': 1.0,
                'client': {
                    'local_epochs': 1,
                    'batch_size': 16,
                    'learning_rate': 0.001
                },
                'communication': {
                    'rounds_per_evaluation': 1,
                    'rounds_per_checkpoint': 1
                }
            },
            'security': {
                'encryption': {
                    'enabled': False
                },
                'differential_privacy': {
                    'enabled': False
                }
            }
        }
    
    def test_fedvae_kd_full_pipeline(self):
        """Test the full FedVAE-KD pipeline."""
        # Create FedVAE-KD system
        fedvae_kd = FederatedVAEKnowledgeDistillation(self.config)
        
        # Build models
        fedvae_kd.build_models()
        
        # Create dummy data
        batch_size = 100
        input_dim = self.config['vae']['input_dim']
        X_train = np.random.random((batch_size, input_dim))
        y_train = np.random.randint(0, 2, (batch_size,))
        
        # Test VAE preprocessing
        X_train_encoded = fedvae_kd.preprocess_data_with_vae(X_train)
        self.assertEqual(X_train_encoded.shape, (batch_size, self.config['vae']['latent_dim']))
        
        # Test federated setup
        client_data = [
            {'x': X_train[:30], 'y': y_train[:30]},
            {'x': X_train[30:60], 'y': y_train[30:60]},
            {'x': X_train[60:], 'y': y_train[60:]}
        ]
        
        fedvae_kd.setup_federated_learning(client_data)
        self.assertEqual(len(fedvae_kd.clients), 3)
        self.assertIsNotNone(fedvae_kd.server)
        
        print("FedVAE-KD integration test passed!")


if __name__ == '__main__':
    unittest.main()