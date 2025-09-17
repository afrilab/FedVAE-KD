import tensorflow as tf
import numpy as np
from typing import List, Dict, Any
import copy


class FederatedClient:
    """
    Federated learning client
    """
    def __init__(self, client_id, model, data, config):
        self.client_id = client_id
        self.model = model
        self.data = data
        self.config = config
        self.local_epochs = config['client']['local_epochs']
        self.batch_size = config['client']['batch_size']
        self.learning_rate = config['client']['learning_rate']
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, global_weights):
        """
        Train the model on local data
        """
        # Set global weights
        self.model.set_weights(global_weights)
        
        # Train locally
        history = self.model.fit(
            self.data['x'],
            self.data['y'],
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            verbose=0
        )
        
        # Get updated weights
        updated_weights = self.model.get_weights()
        
        # Calculate number of samples
        num_samples = len(self.data['x'])
        
        return updated_weights, num_samples


class FederatedServer:
    """
    Federated learning server
    """
    def __init__(self, global_model, config):
        self.global_model = global_model
        self.config = config
        self.num_clients = config['num_clients']
        self.num_rounds = config['num_rounds']
        self.fraction_clients = config['fraction_clients']
        
        # Initialize global model
        self.global_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config['client']['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def aggregate_weights(self, weights_list, sample_counts):
        """
        Federated Averaging (FedAvg) algorithm
        """
        # Normalize sample counts
        total_samples = sum(sample_counts)
        weights_avg = []
        
        # Average weights based on sample counts
        for weights_i in zip(*weights_list):
            weighted_avg = np.average(weights_i, axis=0, weights=sample_counts)
            weights_avg.append(weighted_avg)
        
        return weights_avg
    
    def select_clients(self, clients):
        """
        Select a fraction of clients randomly
        """
        num_selected = max(1, int(len(clients) * self.fraction_clients))
        selected_clients = np.random.choice(clients, num_selected, replace=False)
        return selected_clients
    
    def train_round(self, clients):
        """
        Perform one round of federated training
        """
        # Select clients
        selected_clients = self.select_clients(clients)
        
        # Collect weights and sample counts
        weights_list = []
        sample_counts = []
        
        # Get current global weights
        global_weights = self.global_model.get_weights()
        
        # Train selected clients
        for client in selected_clients:
            weights, num_samples = client.train(global_weights)
            weights_list.append(weights)
            sample_counts.append(num_samples)
        
        # Aggregate weights
        aggregated_weights = self.aggregate_weights(weights_list, sample_counts)
        
        # Update global model
        self.global_model.set_weights(aggregated_weights)
        
        return aggregated_weights
    
    def train(self, clients):
        """
        Train the federated model for multiple rounds
        """
        for round_num in range(self.num_rounds):
            print(f"Round {round_num + 1}/{self.num_rounds}")
            
            # Perform one round of training
            aggregated_weights = self.train_round(clients)
            
            # Evaluate global model (optional)
            if (round_num + 1) % self.config['communication']['rounds_per_evaluation'] == 0:
                # Evaluation code would go here
                pass
            
            # Save checkpoint (optional)
            if (round_num + 1) % self.config['communication']['rounds_per_checkpoint'] == 0:
                # Checkpoint code would go here
                pass
        
        return self.global_model


def create_federated_server(model, config):
    """
    Create federated server
    
    Args:
        model: Global model
        config (dict): Federated learning configuration
    
    Returns:
        FederatedServer: Federated server instance
    """
    return FederatedServer(model, config)


def create_federated_client(client_id, model, data, config):
    """
    Create federated client
    
    Args:
        client_id: Client identifier
        model: Client model
        data: Client data
        config (dict): Federated learning configuration
    
    Returns:
        FederatedClient: Federated client instance
    """
    return FederatedClient(client_id, model, data, config)