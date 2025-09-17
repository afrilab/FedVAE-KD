import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import yaml


def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_dataset(file_path, label_column=None):
    """
    Load dataset from CSV file
    
    Args:
        file_path (str): Path to CSV file
        label_column (str): Name of the label column (if any)
        
    Returns:
        tuple: (features, labels) if label_column is provided, otherwise just features
    """
    # Load data
    data = pd.read_csv(file_path)
    
    if label_column is not None:
        # Separate features and labels
        if label_column in data.columns:
            labels = data[label_column]
            features = data.drop(columns=[label_column])
        else:
            raise ValueError(f"Label column '{label_column}' not found in dataset")
        
        return features.values, labels.values
    else:
        return data.values


def preprocess_data(data, labels=None, normalize=True, encode_labels=True):
    """
    Preprocess data for training
    
    Args:
        data: Input data
        labels: Labels (optional)
        normalize (bool): Whether to normalize features
        encode_labels (bool): Whether to encode labels
        
    Returns:
        Preprocessed data and labels (if provided)
    """
    # Convert to numpy arrays if needed
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # Normalize features
    if normalize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    
    # Encode labels if provided
    if labels is not None:
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        
        if encode_labels and labels.dtype == object:
            le = LabelEncoder()
            labels = le.fit_transform(labels)
        
        return data, labels
    
    return data


def split_data(features, labels, test_size=0.2, validation_size=0.1, random_state=42):
    """
    Split data into train, validation, and test sets
    
    Args:
        features: Input features
        labels: Labels
        test_size (float): Proportion of data for testing
        validation_size (float): Proportion of training data for validation
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: train+validation vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Second split: train vs validation
    if validation_size > 0:
        # Calculate validation size as proportion of train+validation set
        val_proportion = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_proportion, random_state=random_state, stratify=y_train_val
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        return X_train_val, None, X_test, y_train_val, None, y_test


def create_federated_data_splits(features, labels, num_clients, random_state=42):
    """
    Split data for federated learning
    
    Args:
        features: Input features
        labels: Labels
        num_clients (int): Number of clients
        random_state (int): Random seed
        
    Returns:
        list: List of client data dictionaries [{'x': data, 'y': labels}, ...]
    """
    # Shuffle data
    np.random.seed(random_state)
    indices = np.random.permutation(len(features))
    features = features[indices]
    labels = labels[indices]
    
    # Split data among clients
    client_data = []
    samples_per_client = len(features) // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        if i == num_clients - 1:
            # Last client gets remaining samples
            end_idx = len(features)
        else:
            end_idx = (i + 1) * samples_per_client
        
        client_features = features[start_idx:end_idx]
        client_labels = labels[start_idx:end_idx]
        
        client_data.append({
            'x': client_features,
            'y': client_labels
        })
    
    return client_data


def save_results(results, file_path):
    """
    Save results to file
    
    Args:
        results (dict): Results dictionary
        file_path (str): Path to save results
    """
    df = pd.DataFrame([results])
    df.to_csv(file_path, index=False)