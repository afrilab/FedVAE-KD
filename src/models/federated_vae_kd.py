import tensorflow as tf
import numpy as np
from typing import List, Dict, Any
from .vae import VAE, create_vae_model
from .classifier import TeacherModel, StudentModel, create_teacher_model, create_student_model
from ..training.knowledge_distillation import KnowledgeDistillation, create_knowledge_distillation_model
from ..training.federated import FederatedClient, FederatedServer
from ..security.cryptographic_utils import CryptographicUtils, secure_aggregate_weights


class FederatedVAEKnowledgeDistillation:
    """
    Main FedVAE-KD system that combines VAE, Knowledge Distillation, and Federated Learning
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vae = None
        self.teacher = None
        self.student = None
        self.kd_model = None
        self.clients = []
        self.server = None
        self.crypto_utils = CryptographicUtils()
        
        # Security settings
        self.encryption_enabled = config.get('security', {}).get('encryption', {}).get('enabled', False)
        self.differential_privacy_enabled = config.get('security', {}).get('differential_privacy', {}).get('enabled', False)
        self.dp_config = config.get('security', {}).get('differential_privacy', {})
        
        # Initialize encryption key if needed
        self.encryption_key = None
        if self.encryption_enabled:
            password = "fedvae-kd-secret-key"  # In practice, this should be securely managed
            self.encryption_key, _ = self.crypto_utils.generate_key_from_password(password)
    
    def build_models(self):
        """
        Build all models (VAE, Teacher, Student)
        """
        # Create VAE model
        self.vae = create_vae_model(self.config)
        
        # Create Teacher model
        self.teacher = create_teacher_model(self.config)
        
        # Create Student model
        self.student = create_student_model(self.config)
        
        # Create Knowledge Distillation model
        self.kd_model = create_knowledge_distillation_model(
            student=self.student,
            teacher=self.teacher,
            config=self.config
        )
    
    def preprocess_data_with_vae(self, data):
        """
        Preprocess data using VAE encoder
        
        Args:
            data: Input data
            
        Returns:
            Encoded data in latent space
        """
        if self.vae is None:
            raise ValueError("VAE model not initialized. Call build_models() first.")
        
        # Encode data to latent space
        mu, _ = self.vae.encode(data)
        return mu
    
    def train_vae(self, train_data, validation_data=None):
        """
        Train the VAE model
        
        Args:
            train_data: Training data
            validation_data: Validation data (optional)
        """
        if self.vae is None:
            raise ValueError("VAE model not initialized. Call build_models() first.")
        
        # Compile VAE
        self.vae.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.get('training', {}).get('optimizer', {}).get('learning_rate', 0.001)
            ),
            loss=tf.keras.losses.BinaryCrossentropy()
        )
        
        # Train VAE
        history = self.vae.fit(
            train_data,
            train_data,  # Autoencoder reconstruction
            epochs=self.config.get('training', {}).get('epochs', 100),
            batch_size=self.config.get('training', {}).get('batch_size', 128),
            validation_data=(validation_data, validation_data) if validation_data is not None else None,
            verbose=1
        )
        
        return history
    
    def train_teacher_centralized(self, train_data, train_labels, validation_data=None, validation_labels=None):
        """
        Train the teacher model in a centralized manner
        
        Args:
            train_data: Training data (in latent space)
            train_labels: Training labels
            validation_data: Validation data (optional)
            validation_labels: Validation labels (optional)
        """
        if self.teacher is None:
            raise ValueError("Teacher model not initialized. Call build_models() first.")
        
        # Compile teacher model
        self.teacher.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.get('training', {}).get('optimizer', {}).get('learning_rate', 0.001)
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train teacher model
        history = self.teacher.fit(
            train_data,
            train_labels,
            epochs=self.config.get('training', {}).get('epochs', 100),
            batch_size=self.config.get('training', {}).get('batch_size', 128),
            validation_data=(validation_data, validation_labels) if validation_data is not None else None,
            verbose=1
        )
        
        return history
    
    def train_student_with_kd(self, train_data, train_labels, validation_data=None, validation_labels=None):
        """
        Train the student model using knowledge distillation
        
        Args:
            train_data: Training data (in latent space)
            train_labels: Training labels
            validation_data: Validation data (optional)
            validation_labels: Validation labels (optional)
        """
        if self.kd_model is None:
            raise ValueError("Knowledge distillation model not initialized. Call build_models() first.")
        
        # Compile KD model
        self.kd_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.get('training', {}).get('optimizer', {}).get('learning_rate', 0.001)
            ),
            metrics=['accuracy'],
            student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
            distillation_loss_fn=tf.keras.losses.KLDivergence()
        )
        
        # Train student model with knowledge distillation
        history = self.kd_model.fit(
            train_data,
            train_labels,
            epochs=self.config.get('training', {}).get('epochs', 100),
            batch_size=self.config.get('training', {}).get('batch_size', 128),
            validation_data=(validation_data, validation_labels) if validation_data is not None else None,
            verbose=1
        )
        
        return history
    
    def setup_federated_learning(self, client_data):
        """
        Set up federated learning environment
        
        Args:
            client_data: List of client data dictionaries [{'x': data, 'y': labels}, ...]
        """
        if self.student is None:
            raise ValueError("Student model not initialized. Call build_models() first.")
        
        # Create federated clients
        self.clients = []
        for i, data in enumerate(client_data):
            client_model = create_student_model(self.config)  # Each client gets a copy of the student model
            client = FederatedClient(
                client_id=i,
                model=client_model,
                data=data,
                config=self.config.get('federated', {})
            )
            self.clients.append(client)
        
        # Create federated server
        self.server = FederatedServer(
            global_model=self.student,  # Global model is the student model
            config=self.config.get('federated', {})
        )
    
    def train_federated_kd(self):
        """
        Train the student model using federated knowledge distillation
        """
        if self.server is None or len(self.clients) == 0:
            raise ValueError("Federated learning not set up. Call setup_federated_learning() first.")
        
        # Train using federated learning
        trained_model = self.server.train(self.clients)
        return trained_model
    
    def secure_aggregate(self, weights_list, sample_counts):
        """
        Securely aggregate weights from clients
        
        Args:
            weights_list: List of weights from clients
            sample_counts: Number of samples for each client
            
        Returns:
            Aggregated weights
        """
        return secure_aggregate_weights(
            weights_list=weights_list,
            sample_counts=sample_counts,
            encryption_key=self.encryption_key if self.encryption_enabled else None,
            differential_privacy=self.differential_privacy_enabled,
            dp_config=self.dp_config if self.differential_privacy_enabled else None
        )
    
    def evaluate(self, test_data, test_labels):
        """
        Evaluate the student model
        
        Args:
            test_data: Test data (in latent space)
            test_labels: Test labels
            
        Returns:
            Evaluation metrics
        """
        if self.student is None:
            raise ValueError("Student model not initialized.")
        
        # Compile model for evaluation
        self.student.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Evaluate model
        results = self.student.evaluate(test_data, test_labels, verbose=0)
        return dict(zip(self.student.metrics_names, results))


def create_fedvae_kd_system(config):
    """
    Create FedVAE-KD system
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        FederatedVAEKnowledgeDistillation: FedVAE-KD system instance
    """
    return FederatedVAEKnowledgeDistillation(config)