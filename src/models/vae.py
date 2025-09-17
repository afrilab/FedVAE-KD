import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class KLDivergenceLayer(layers.Layer):
    """
    Custom Keras layer for computing KL divergence in VAE
    """
    def __init__(self, beta=1.0, **kwargs):
        super(KLDivergenceLayer, self).__init__(**kwargs)
        self.beta = beta
        self.is_placeholder = True

    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=-1)
        self.add_loss(self.beta * tf.reduce_mean(kl_batch), inputs=inputs)
        return inputs


class VAE(keras.Model):
    """
    Variational Autoencoder implementation for network intrusion detection
    """
    def __init__(self, input_dim, latent_dim, hidden_layers, dropout_rate=0.2, beta=1.0):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.beta = beta
        
        # Encoder
        self.encoder_layers = []
        for units in hidden_layers:
            self.encoder_layers.extend([
                layers.Dense(units, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(dropout_rate)
            ])
        
        self.encoder_dense = layers.Dense(latent_dim * 2)  # For mu and log_var
        self.kl_divergence = KLDivergenceLayer(beta=beta)
        
        # Decoder
        self.decoder_layers = []
        for units in reversed(hidden_layers):
            self.decoder_layers.extend([
                layers.Dense(units, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(dropout_rate)
            ])
        
        self.decoder_output = layers.Dense(input_dim, activation='sigmoid')
    
    def encode(self, x):
        """Encode input to latent space"""
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.encoder_dense(x)
        mu, log_var = tf.split(x, num_or_size_splits=2, axis=1)
        mu, log_var = self.kl_divergence([mu, log_var])
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick"""
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var * 0.5) * eps
    
    def decode(self, z):
        """Decode latent vector to reconstruction"""
        for layer in self.decoder_layers:
            z = layer(z)
        return self.decoder_output(z)
    
    def call(self, x, training=None):
        """Forward pass"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed
    
    def sample(self, num_samples=1):
        """Generate samples from the latent space"""
        z = tf.random.normal(shape=(num_samples, self.latent_dim))
        return self.decode(z)
    
    def reconstruct(self, x):
        """Reconstruct input"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z)


def create_vae_model(config):
    """
    Create VAE model from configuration
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        VAE: VAE model instance
    """
    vae_config = config['vae']
    return VAE(
        input_dim=vae_config['input_dim'],
        latent_dim=vae_config['latent_dim'],
        hidden_layers=vae_config['hidden_layers'],
        dropout_rate=vae_config['dropout_rate'],
        beta=1.0
    )