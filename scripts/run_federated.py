#!/usr/bin/env python3
"""
Federated training script for FedVAE-KD
"""
import argparse
import os
import sys
import numpy as np
import tensorflow as tf

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.data_utils import load_config, load_dataset, preprocess_data, split_data, create_federated_data_splits
from src.models.federated_vae_kd import create_fedvae_kd_system


def main():
    parser = argparse.ArgumentParser(description='Run federated training for FedVAE-KD')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset')
    parser.add_argument('--label-column', type=str, default='label', help='Name of label column')
    parser.add_argument('--algorithm', type=str, default='FedAvg', choices=['FedAvg', 'FedProx', 'FedAdam'], 
                        help='Federated learning algorithm')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update federated config with selected algorithm
    config['federated']['aggregation']['selected'] = args.algorithm
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    features, labels = load_dataset(args.data, label_column=args.label_column)
    features, labels = preprocess_data(features, labels)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        features, labels, 
        test_size=0.2, 
        validation_size=0.1
    )
    
    # Create federated data splits
    num_clients = config['federated']['num_clients']
    client_data = create_federated_data_splits(X_train, y_train, num_clients)
    
    # Create FedVAE-KD system
    print("Creating FedVAE-KD system...")
    fedvae_kd = create_fedvae_kd_system(config)
    fedvae_kd.build_models()
    
    # Train VAE (centralized)
    print("Training VAE...")
    vae_history = fedvae_kd.train_vae(X_train, X_val)
    
    # Encode data using VAE
    print("Encoding data with VAE...")
    X_train_encoded = fedvae_kd.preprocess_data_with_vae(X_train)
    X_val_encoded = fedvae_kd.preprocess_data_with_vae(X_val) if X_val is not None else None
    X_test_encoded = fedvae_kd.preprocess_data_with_vae(X_test)
    
    # Update client data with encoded features
    for i, client in enumerate(client_data):
        start_idx = i * len(X_train_encoded) // num_clients
        if i == num_clients - 1:
            end_idx = len(X_train_encoded)
        else:
            end_idx = (i + 1) * len(X_train_encoded) // num_clients
            
        client['x'] = X_train_encoded[start_idx:end_idx]
        client['y'] = y_train[start_idx:end_idx]
    
    # Train teacher model (centralized)
    print("Training teacher model...")
    teacher_history = fedvae_kd.train_teacher_centralized(
        X_train_encoded, y_train,
        X_val_encoded, y_val
    )
    
    # Set up federated learning for student model
    print("Setting up federated learning...")
    fedvae_kd.setup_federated_learning(client_data)
    
    # Train student model using federated learning
    print(f"Training student model using {args.algorithm}...")
    trained_model = fedvae_kd.train_federated_kd()
    
    # Evaluate student model
    print("Evaluating student model...")
    test_metrics = fedvae_kd.evaluate(X_test_encoded, y_test)
    print(f"Test metrics: {test_metrics}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save metrics
    import pandas as pd
    results_df = pd.DataFrame([test_metrics])
    results_df.to_csv(os.path.join(args.output_dir, f'federated_{args.algorithm}_results.csv'), index=False)
    
    # Save models
    fedvae_kd.vae.save(os.path.join(args.output_dir, 'vae_model'))
    fedvae_kd.teacher.save(os.path.join(args.output_dir, 'teacher_model'))
    fedvae_kd.student.save(os.path.join(args.output_dir, f'student_model_{args.algorithm}'))
    
    print(f"Federated training with {args.algorithm} completed!")


if __name__ == '__main__':
    main()