import unittest
import numpy as np
import tensorflow as tf
from src.models.classifier import TeacherModel, StudentModel, create_teacher_model, create_student_model


class TestClassifier(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.input_dim = 10
        self.hidden_layers = [20, 15]
        self.output_dim = 3
        self.dropout_rate = 0.1
        
        self.teacher = TeacherModel(self.input_dim, self.hidden_layers, self.output_dim, self.dropout_rate)
        self.student = StudentModel(self.input_dim, self.hidden_layers, self.output_dim, self.dropout_rate)
    
    def test_teacher_initialization(self):
        """Test TeacherModel initialization."""
        self.assertEqual(self.teacher.input_dim, self.input_dim)
        self.assertEqual(self.teacher.hidden_layers, self.hidden_layers)
        self.assertEqual(self.teacher.output_dim, self.output_dim)
        self.assertEqual(self.teacher.dropout_rate, self.dropout_rate)
    
    def test_student_initialization(self):
        """Test StudentModel initialization."""
        self.assertEqual(self.student.input_dim, self.input_dim)
        self.assertEqual(self.student.hidden_layers, self.hidden_layers)
        self.assertEqual(self.student.output_dim, self.output_dim)
        self.assertEqual(self.student.dropout_rate, self.dropout_rate)
    
    def test_teacher_call(self):
        """Test TeacherModel forward pass."""
        batch_size = 32
        x = tf.random.normal((batch_size, self.input_dim))
        
        output = self.teacher(x)
        
        self.assertEqual(output.shape, (batch_size, self.output_dim))
    
    def test_student_call(self):
        """Test StudentModel forward pass."""
        batch_size = 32
        x = tf.random.normal((batch_size, self.input_dim))
        
        output = self.student(x)
        
        self.assertEqual(output.shape, (batch_size, self.output_dim))
    
    def test_create_teacher_model(self):
        """Test TeacherModel creation from config."""
        config = {
            'teacher': {
                'input_dim': 20,
                'hidden_layers': [32, 16],
                'output_dim': 5,
                'dropout_rate': 0.2
            }
        }
        
        teacher_model = create_teacher_model(config)
        
        self.assertIsInstance(teacher_model, TeacherModel)
        self.assertEqual(teacher_model.input_dim, 20)
        self.assertEqual(teacher_model.output_dim, 5)
    
    def test_create_student_model(self):
        """Test StudentModel creation from config."""
        config = {
            'student': {
                'input_dim': 20,
                'hidden_layers': [32, 16],
                'output_dim': 5,
                'dropout_rate': 0.2
            }
        }
        
        student_model = create_student_model(config)
        
        self.assertIsInstance(student_model, StudentModel)
        self.assertEqual(student_model.input_dim, 20)
        self.assertEqual(student_model.output_dim, 5)


if __name__ == '__main__':
    unittest.main()