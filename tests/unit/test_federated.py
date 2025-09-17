import unittest
import numpy as np
import tensorflow as tf
from src.training.federated import FederatedClient, FederatedServer


class TestFederated(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a simple model for testing
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        # Create dummy data
        self.data = {
            'x': np.random.random((100, 5)),
            'y': np.random.randint(0, 3, (100,))
        }
        
        # Create config
        self.config = {
            'client': {
                'local_epochs': 2,
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
    
    def test_federated_client_initialization(self):
        """Test FederatedClient initialization."""
        client = FederatedClient(1, self.model, self.data, self.config)
        
        self.assertEqual(client.client_id, 1)
        self.assertEqual(client.local_epochs, 2)
        self.assertEqual(client.batch_size, 32)
        self.assertEqual(client.learning_rate, 0.001)
    
    def test_federated_client_train(self):
        """Test FederatedClient training."""
        client = FederatedClient(1, self.model, self.data, self.config)
        
        # Get initial weights
        initial_weights = client.model.get_weights()
        
        # Train client
        updated_weights, num_samples = client.train(initial_weights)
        
        # Check that weights were updated
        self.assertNotEqual(initial_weights[0][0, 0], updated_weights[0][0, 0])
        self.assertEqual(num_samples, 100)
    
    def test_federated_server_initialization(self):
        """Test FederatedServer initialization."""
        server_config = {
            'num_clients': 10,
            'num_rounds': 5,
            'fraction_clients': 0.5,
            'client': {
                'local_epochs': 2,
                'batch_size': 32,
                'learning_rate': 0.001
            },
            'communication': {
                'rounds_per_evaluation': 1,
                'rounds_per_checkpoint': 2
            }
        }
        
        server = FederatedServer(self.model, server_config)
        
        self.assertEqual(server.num_clients, 10)
        self.assertEqual(server.num_rounds, 5)
        self.assertEqual(server.fraction_clients, 0.5)
    
    def test_federated_server_aggregate_weights(self):
        """Test FederatedServer weight aggregation."""
        server_config = {
            'num_clients': 10,
            'num_rounds': 5,
            'fraction_clients': 0.5,
            'client': {
                'local_epochs': 2,
                'batch_size': 32,
                'learning_rate': 0.001
            },
            'communication': {
                'rounds_per_evaluation': 1,
                'rounds_per_checkpoint': 2
            }
        }
        
        server = FederatedServer(self.model, server_config)
        
        # Create dummy weights
        weights1 = [np.random.random((5, 10)), np.random.random((10,))]
        weights2 = [np.random.random((5, 10)), np.random.random((10,))]
        weights_list = [weights1, weights2]
        sample_counts = [50, 50]
        
        # Aggregate weights
        aggregated_weights = server.aggregate_weights(weights_list, sample_counts)
        
        # Check aggregation
        expected_weights = [(weights1[0] + weights2[0]) / 2, (weights1[1] + weights2[1]) / 2]
        
        np.testing.assert_array_almost_equal(aggregated_weights[0], expected_weights[0])
        np.testing.assert_array_almost_equal(aggregated_weights[1], expected_weights[1])


if __name__ == '__main__':
    unittest.main()