import tensorflow as tf
from tensorflow import keras
import numpy as np


class KnowledgeDistillation(keras.Model):
    """
    Knowledge Distillation framework
    """
    def __init__(self, student, teacher, temperature=3.0, alpha=0.5):
        super(KnowledgeDistillation, self).__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher model
        self.teacher.trainable = False
    
    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn):
        """
        Configure the model for training
        """
        super(KnowledgeDistillation, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
    
    def train_step(self, data):
        """
        Custom training step for knowledge distillation
        """
        x, y = data
        
        # Get teacher predictions
        teacher_predictions = self.teacher(x, training=False)
        
        with tf.GradientTape() as tape:
            # Get student predictions
            student_predictions = self.student(x, training=True)
            
            # Calculate student loss
            student_loss = self.student_loss_fn(y, student_predictions)
            
            # Calculate distillation loss
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1)
            )
            
            # Combine losses
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        
        # Calculate gradients and update weights
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.compiled_metrics.update_state(y, student_predictions)
        
        # Return metrics
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "student_loss": student_loss,
            "distillation_loss": distillation_loss
        })
        return results
    
    def test_step(self, data):
        """
        Custom test step
        """
        x, y = data
        
        # Get student predictions
        student_predictions = self.student(x, training=False)
        
        # Calculate student loss
        student_loss = self.student_loss_fn(y, student_predictions)
        
        # Update metrics
        self.compiled_metrics.update_state(y, student_predictions)
        
        # Return metrics
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results
    
    def call(self, inputs):
        """
        Forward pass
        """
        return self.student(inputs)


def create_knowledge_distillation_model(student, teacher, config):
    """
    Create knowledge distillation model
    
    Args:
        student: Student model
        teacher: Teacher model
        config (dict): Configuration dictionary
    
    Returns:
        KnowledgeDistillation: Knowledge distillation model
    """
    # Extract KD parameters from config (using default values if not present)
    temperature = config.get('knowledge_distillation', {}).get('temperature', 3.0)
    alpha = config.get('knowledge_distillation', {}).get('alpha', 0.5)
    
    return KnowledgeDistillation(
        student=student,
        teacher=teacher,
        temperature=temperature,
        alpha=alpha
    )


def distillation_loss_fn(teacher_predictions, student_predictions):
    """
    Distillation loss function (KL divergence)
    """
    return tf.keras.losses.KLDivergence()(teacher_predictions, student_predictions)