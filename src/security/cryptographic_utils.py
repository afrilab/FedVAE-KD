import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import tensorflow as tf
import numpy as np


class CryptographicUtils:
    """
    Cryptographic utilities for secure federated learning
    """
    
    @staticmethod
    def generate_key_from_password(password: str, salt: bytes = None) -> tuple:
        """
        Generate a key from a password using PBKDF2
        
        Args:
            password (str): Password to derive key from
            salt (bytes): Salt for key derivation (randomly generated if None)
        
        Returns:
            tuple: (key, salt)
        """
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt
    
    @staticmethod
    def encrypt_weights(weights, key) -> bytes:
        """
        Encrypt model weights using Fernet symmetric encryption
        
        Args:
            weights (list): Model weights as numpy arrays
            key (bytes): Encryption key
        
        Returns:
            bytes: Encrypted weights
        """
        f = Fernet(key)
        
        # Serialize weights
        weights_serialized = [w.tolist() for w in weights]
        weights_str = str(weights_serialized)
        
        # Encrypt
        encrypted_weights = f.encrypt(weights_str.encode())
        return encrypted_weights
    
    @staticmethod
    def decrypt_weights(encrypted_weights, key) -> list:
        """
        Decrypt model weights using Fernet symmetric encryption
        
        Args:
            encrypted_weights (bytes): Encrypted weights
            key (bytes): Decryption key
        
        Returns:
            list: Decrypted weights as numpy arrays
        """
        f = Fernet(key)
        
        # Decrypt
        weights_str = f.decrypt(encrypted_weights).decode()
        weights_serialized = eval(weights_str)
        
        # Deserialize weights
        weights = [np.array(w) for w in weights_serialized]
        return weights
    
    @staticmethod
    def add_differential_privacy_noise(weights, noise_multiplier=1.0, l2_norm_clip=1.0) -> list:
        """
        Add Gaussian noise to weights for differential privacy
        
        Args:
            weights (list): Model weights
            noise_multiplier (float): Noise multiplier for privacy
            l2_norm_clip (float): L2 norm clipping threshold
        
        Returns:
            list: Noised weights
        """
        noised_weights = []
        
        for w in weights:
            # Clip weights to L2 norm
            l2_norm = np.linalg.norm(w)
            if l2_norm > l2_norm_clip:
                w = w * (l2_norm_clip / l2_norm)
            
            # Add Gaussian noise
            noise = np.random.normal(0, noise_multiplier * l2_norm_clip, w.shape)
            noised_w = w + noise
            noised_weights.append(noised_w)
        
        return noised_weights


def secure_aggregate_weights(weights_list, sample_counts, encryption_key=None, 
                           differential_privacy=False, dp_config=None) -> list:
    """
    Securely aggregate weights from multiple clients
    
    Args:
        weights_list (list): List of weights from clients
        sample_counts (list): Number of samples for each client
        encryption_key (bytes): Encryption key for secure aggregation
        differential_privacy (bool): Whether to apply differential privacy
        dp_config (dict): Differential privacy configuration
    
    Returns:
        list: Aggregated weights
    """
    # Decrypt weights if encryption is used
    if encryption_key is not None:
        crypto_utils = CryptographicUtils()
        decrypted_weights_list = []
        for encrypted_weights in weights_list:
            decrypted_weights = crypto_utils.decrypt_weights(encrypted_weights, encryption_key)
            decrypted_weights_list.append(decrypted_weights)
        weights_list = decrypted_weights_list
    
    # Normalize sample counts
    total_samples = sum(sample_counts)
    normalized_counts = [count / total_samples for count in sample_counts]
    
    # Aggregate weights
    aggregated_weights = []
    for weights_i in zip(*weights_list):
        weighted_avg = np.average(weights_i, axis=0, weights=normalized_counts)
        aggregated_weights.append(weighted_avg)
    
    # Apply differential privacy if enabled
    if differential_privacy and dp_config is not None:
        crypto_utils = CryptographicUtils()
        aggregated_weights = crypto_utils.add_differential_privacy_noise(
            aggregated_weights,
            noise_multiplier=dp_config.get('noise_multiplier', 1.0),
            l2_norm_clip=dp_config.get('max_grad_norm', 1.0)
        )
    
    # Encrypt aggregated weights if encryption is used
    if encryption_key is not None:
        crypto_utils = CryptographicUtils()
        encrypted_aggregated_weights = crypto_utils.encrypt_weights(aggregated_weights, encryption_key)
        return encrypted_aggregated_weights
    
    return aggregated_weights