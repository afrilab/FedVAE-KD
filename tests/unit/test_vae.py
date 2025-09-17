import unittest
import numpy as np
import tensorflow as tf
from src.models.vae import VAE, create_vae_model


class TestVAE(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.input_dim = 10
        self.latent_dim = 5
        self.hidden_layers = [20, 15]
        self.dropout_rate = 0.1
        self.vae = VAE(self.input_dim, self.latent_dim, self.hidden_layers, self.dropout_rate)
    
    def test_vae_initialization(self):
        """Test VAE initialization."""
        self.assertEqual(self.vae.input_dim, self.input_dim)
        self.assertEqual(self.vae.latent_dim, self.latent_dim)
        self.assertEqual(self.vae.hidden_layers, self.hidden_layers)
        self.assertEqual(self.vae.dropout_rate, self.dropout_rate)
    
    def test_vae_encode(self):
        """Test VAE encoding functionality."""
        batch_size = 32
        x = tf.random.normal((batch_size, self.input_dim))
        
        mu, log_var = self.vae.encode(x)
        
        self.assertEqual(mu.shape, (batch_size, self.latent_dim))
        self.assertEqual(log_var.shape, (batch_size, self.latent_dim))
    
    def test_vae_reparameterize(self):
        """Test VAE reparameterization functionality."""
        batch_size = 32
        mu = tf.random.normal((batch_size, self.latent_dim))
        log_var = tf.random.normal((batch_size, self.latent_dim))
        
        z = self.vae.reparameterize(mu, log_var)
        
        self.assertEqual(z.shape, (batch_size, self.latent_dim))
    
    def test_vae_decode(self):
        """Test VAE decoding functionality."""
        batch_size = 32
        z = tf.random.normal((batch_size, self.latent_dim))
        
        reconstructed = self.vae.decode(z)
        
        self.assertEqual(reconstructed.shape, (batch_size, self.input_dim))
    
    def test_vae_call(self):
        """Test VAE forward pass."""
        batch_size = 32
        x = tf.random.normal((batch_size, self.input_dim))
        
        reconstructed = self.vae(x)
        
        self.assertEqual(reconstructed.shape, (batch_size, self.input_dim))
    
    def test_vae_sample(self):
        """Test VAE sampling functionality."""
        num_samples = 10
        samples = self.vae.sample(num_samples)
        
        self.assertEqual(samples.shape, (num_samples, self.input_dim))
    
    def test_create_vae_model(self):
        """Test VAE model creation from config."""
        config = {
            'vae': {
                'input_dim': 20,
                'latent_dim': 8,
                'hidden_layers': [32, 16],
                'dropout_rate': 0.2
            }
        }
        
        vae_model = create_vae_model(config)
        
        self.assertIsInstance(vae_model, VAE)
        self.assertEqual(vae_model.input_dim, 20)
        self.assertEqual(vae_model.latent_dim, 8)


if __name__ == '__main__':
    unittest.main()