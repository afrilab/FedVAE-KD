import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TeacherModel(keras.Model):
    """
    Teacher model for knowledge distillation
    """
    def __init__(self, input_dim, hidden_layers, output_dim, dropout_rate=0.3):
        super(TeacherModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Define layers
        self.hidden_layers_list = []
        for units in hidden_layers:
            self.hidden_layers_list.extend([
                layers.Dense(units, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(dropout_rate)
            ])
        
        self.output_layer = layers.Dense(output_dim, activation='softmax')
    
    def call(self, inputs, training=None):
        x = inputs
        for layer in self.hidden_layers_list:
            x = layer(x)
        return self.output_layer(x)


class StudentModel(keras.Model):
    """
    Student model for knowledge distillation
    """
    def __init__(self, input_dim, hidden_layers, output_dim, dropout_rate=0.2):
        super(StudentModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Define layers
        self.hidden_layers_list = []
        for units in hidden_layers:
            self.hidden_layers_list.extend([
                layers.Dense(units, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(dropout_rate)
            ])
        
        self.output_layer = layers.Dense(output_dim, activation='softmax')
    
    def call(self, inputs, training=None):
        x = inputs
        for layer in self.hidden_layers_list:
            x = layer(x)
        return self.output_layer(x)


def create_teacher_model(config):
    """
    Create teacher model from configuration
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        TeacherModel: Teacher model instance
    """
    teacher_config = config['teacher']
    return TeacherModel(
        input_dim=teacher_config['input_dim'],
        hidden_layers=teacher_config['hidden_layers'],
        output_dim=teacher_config['output_dim'],
        dropout_rate=teacher_config['dropout_rate']
    )


def create_student_model(config):
    """
    Create student model from configuration
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        StudentModel: Student model instance
    """
    student_config = config['student']
    return StudentModel(
        input_dim=student_config['input_dim'],
        hidden_layers=student_config['hidden_layers'],
        output_dim=student_config['output_dim'],
        dropout_rate=student_config['dropout_rate']
    )