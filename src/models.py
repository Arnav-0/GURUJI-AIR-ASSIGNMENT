"""
Machine Learning Models Module
Contains model architectures and training utilities
"""

import numpy as np
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib


def create_cnn_model(
    input_shape: Tuple[int, int] = (64, 64),
    num_classes: int = 2
) -> keras.Model:
    """
    Create CNN model for radar heatmap classification
    
    Args:
        input_shape: Input image dimensions
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(*input_shape, 1)),
        
        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    # Use categorical crossentropy for 2 classes with one-hot encoding
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model


def create_lightweight_cnn(
    input_shape: Tuple[int, int] = (64, 64)
) -> keras.Model:
    """
    Create lightweight CNN for embedded deployment
    
    Args:
        input_shape: Input image dimensions
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        layers.Input(shape=(*input_shape, 1)),
        
        # Depthwise separable convolutions for efficiency
        layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


class SVMClassifier:
    """
    SVM-based classifier for radar data
    """
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0):
        """
        Initialize SVM classifier
        
        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly')
            C: Regularization parameter
        """
        self.svm = SVC(kernel=kernel, C=C, probability=True, random_state=42)
        self.scaler = StandardScaler()
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ):
        """
        Train SVM classifier
        
        Args:
            X_train: Training features (flattened)
            y_train: Training labels
        """
        # Flatten if needed
        if len(X_train.shape) > 2:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
        else:
            X_train_flat = X_train
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train_flat)
        
        # Train SVM
        self.svm.fit(X_scaled, y_train)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict labels
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted labels
        """
        # Flatten if needed
        if len(X_test.shape) > 2:
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
        else:
            X_test_flat = X_test
        
        # Scale and predict
        X_scaled = self.scaler.transform(X_test_flat)
        return self.svm.predict(X_scaled)
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X_test: Test features
            
        Returns:
            Class probabilities
        """
        # Flatten if needed
        if len(X_test.shape) > 2:
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
        else:
            X_test_flat = X_test
        
        # Scale and predict
        X_scaled = self.scaler.transform(X_test_flat)
        return self.svm.predict_proba(X_scaled)
    
    def save(self, filepath: str):
        """Save model to disk"""
        joblib.dump({'svm': self.svm, 'scaler': self.scaler}, filepath)
    
    def load(self, filepath: str):
        """Load model from disk"""
        data = joblib.load(filepath)
        self.svm = data['svm']
        self.scaler = data['scaler']


def prepare_data_for_training(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'cnn'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for model training
    
    Args:
        X: Features
        y: Labels
        model_type: 'cnn' or 'svm'
        
    Returns:
        Processed X and y
    """
    if model_type == 'cnn':
        # Add channel dimension for CNN
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=-1)
        # Ensure values in [0, 1]
        X = np.clip(X, 0, 1)
        # Convert labels to one-hot encoding for categorical crossentropy
        if len(y.shape) == 1:
            from tensorflow.keras.utils import to_categorical
            y = to_categorical(y, num_classes=2)
    elif model_type == 'svm':
        # Flatten for SVM
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
    
    return X, y


def create_callbacks(
    model_path: str,
    patience: int = 10
) -> list:
    """
    Create training callbacks
    
    Args:
        model_path: Path to save best model
        patience: Early stopping patience
        
    Returns:
        List of callbacks
    """
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = 'cnn'
) -> dict:
    """
    Evaluate model and return metrics
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_type: 'cnn' or 'svm'
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, roc_auc_score, roc_curve
    )
    
    # Convert one-hot encoded y_test back to labels if needed
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test_labels = np.argmax(y_test, axis=1)
    else:
        y_test_labels = y_test
    
    # Get predictions
    if model_type == 'cnn':
        y_pred_proba = model.predict(X_test, verbose=0)
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_pred_proba = y_pred_proba[:, 1]
        else:
            y_pred_proba = y_pred_proba.flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
    else:  # SVM
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test_labels, y_pred),
        'precision': precision_score(y_test_labels, y_pred, zero_division=0),
        'recall': recall_score(y_test_labels, y_pred, zero_division=0),
        'f1_score': f1_score(y_test_labels, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test_labels, y_pred),
        'auc': roc_auc_score(y_test_labels, y_pred_proba),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test_labels, y_pred_proba)
    metrics['fpr'] = fpr
    metrics['tpr'] = tpr
    
    return metrics


def print_evaluation_results(metrics: dict):
    """
    Print evaluation metrics
    
    Args:
        metrics: Dictionary of metrics from evaluate_model
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"AUC-ROC:   {metrics['auc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("="*50 + "\n")
