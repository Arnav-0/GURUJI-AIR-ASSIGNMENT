"""
Data Generator Module
Creates synthetic datasets for training and evaluation
"""

import numpy as np
import os
from typing import Tuple, List, Optional
try:
    from .signal_processing import RadarSimulator, normalize_heatmap
except ImportError:
    from signal_processing import RadarSimulator, normalize_heatmap


class SyntheticDatasetGenerator:
    """
    Generates synthetic radar datasets for classification
    """
    
    def __init__(
        self,
        radar_sim: Optional[RadarSimulator] = None,
        output_size: Tuple[int, int] = (64, 64)
    ):
        """
        Initialize dataset generator
        
        Args:
            radar_sim: RadarSimulator instance
            output_size: Output heatmap dimensions
        """
        self.radar_sim = radar_sim if radar_sim else RadarSimulator()
        self.output_size = output_size
        
    def generate_metal_sample(self) -> Tuple[np.ndarray, int]:
        """
        Generate a single metal object sample
        
        Returns:
            Heatmap and label (1 for metal)
        """
        # Random metal object parameters
        range_m = np.random.uniform(1.0, 4.5)
        velocity = np.random.uniform(-1.5, 1.5)
        rcs = np.random.uniform(0.8, 1.5)  # Strong reflection
        noise = np.random.uniform(0.005, 0.02)
        
        # Generate signal
        signal_matrix = self.radar_sim.generate_metal_object(
            range_m=range_m,
            velocity=velocity,
            rcs=rcs,
            noise_power=noise
        )
        
        # Apply windowing
        signal_matrix = self.radar_sim.apply_windowing(signal_matrix, 'hamming')
        
        # Generate range-Doppler map
        rd_map = self.radar_sim.generate_range_doppler_map(signal_matrix)
        
        # Resize and normalize
        heatmap = self._resize_and_normalize(rd_map)
        
        return heatmap, 1
    
    def generate_nonmetal_sample(self) -> Tuple[np.ndarray, int]:
        """
        Generate a single non-metal object sample
        
        Returns:
            Heatmap and label (0 for non-metal)
        """
        sample_type = np.random.choice(['empty', 'weak_reflection', 'clutter'])
        
        if sample_type == 'empty':
            # Empty room with just noise
            noise = np.random.uniform(0.01, 0.03)
            signal_matrix = self.radar_sim.generate_empty_room(noise_power=noise)
            
        elif sample_type == 'weak_reflection':
            # Weak reflecting object (plastic, wood, etc.)
            range_m = np.random.uniform(1.0, 4.5)
            velocity = np.random.uniform(-1.0, 1.0)
            rcs = np.random.uniform(0.1, 0.4)  # Weak reflection
            noise = np.random.uniform(0.01, 0.03)
            
            signal_matrix = self.radar_sim.generate_target_signal(
                ranges=np.array([range_m]),
                velocities=np.array([velocity]),
                rcs=np.array([rcs]),
                noise_power=noise
            )
        else:
            # Multiple weak reflections (clutter)
            num_targets = np.random.randint(2, 4)
            ranges = np.random.uniform(0.5, 5.0, num_targets)
            velocities = np.random.uniform(-1.0, 1.0, num_targets)
            rcs = np.random.uniform(0.1, 0.3, num_targets)
            noise = np.random.uniform(0.015, 0.03)
            
            signal_matrix = self.radar_sim.generate_target_signal(
                ranges=ranges,
                velocities=velocities,
                rcs=rcs,
                noise_power=noise
            )
        
        # Apply windowing
        signal_matrix = self.radar_sim.apply_windowing(signal_matrix, 'hamming')
        
        # Generate range-Doppler map
        rd_map = self.radar_sim.generate_range_doppler_map(signal_matrix)
        
        # Resize and normalize
        heatmap = self._resize_and_normalize(rd_map)
        
        return heatmap, 0
    
    def generate_cluttered_metal_sample(
        self,
        hidden: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Generate sample with metal hidden in clutter
        
        Args:
            hidden: Whether metal is partially obscured
            
        Returns:
            Heatmap and label
        """
        # Metal target
        metal_range = np.random.uniform(1.5, 4.0)
        metal_velocity = np.random.uniform(-1.0, 1.0)
        metal_rcs = np.random.uniform(0.7, 1.2) if not hidden else np.random.uniform(0.5, 0.8)
        
        # Clutter targets
        num_clutter = np.random.randint(3, 6)
        clutter_ranges = np.random.uniform(0.5, 5.0, num_clutter)
        clutter_velocities = np.random.uniform(-1.5, 1.5, num_clutter)
        clutter_rcs = np.random.uniform(0.2, 0.6, num_clutter)
        
        # Combine all targets
        all_ranges = np.concatenate([[metal_range], clutter_ranges])
        all_velocities = np.concatenate([[metal_velocity], clutter_velocities])
        all_rcs = np.concatenate([[metal_rcs], clutter_rcs])
        
        noise = np.random.uniform(0.02, 0.04)
        
        signal_matrix = self.radar_sim.generate_target_signal(
            ranges=all_ranges,
            velocities=all_velocities,
            rcs=all_rcs,
            noise_power=noise
        )
        
        # Apply windowing
        signal_matrix = self.radar_sim.apply_windowing(signal_matrix, 'hamming')
        
        # Generate range-Doppler map
        rd_map = self.radar_sim.generate_range_doppler_map(signal_matrix)
        
        # Resize and normalize
        heatmap = self._resize_and_normalize(rd_map)
        
        return heatmap, 1
    
    def _resize_and_normalize(self, rd_map: np.ndarray) -> np.ndarray:
        """Resize and normalize heatmap"""
        from scipy.ndimage import zoom
        
        # Calculate zoom factors
        zoom_factors = (
            self.output_size[0] / rd_map.shape[0],
            self.output_size[1] / rd_map.shape[1]
        )
        
        # Resize
        resized = zoom(rd_map, zoom_factors, order=1)
        
        # Normalize
        normalized = normalize_heatmap(resized, method='minmax')
        
        return normalized
    
    def generate_dataset(
        self,
        num_samples: int = 300,
        metal_ratio: float = 0.5,
        include_clutter: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete dataset
        
        Args:
            num_samples: Total number of samples
            metal_ratio: Ratio of metal samples
            include_clutter: Include cluttered scenarios
            
        Returns:
            X: Array of heatmaps (N, H, W)
            y: Array of labels (N,)
        """
        num_metal = int(num_samples * metal_ratio)
        num_nonmetal = num_samples - num_metal
        
        X = []
        y = []
        
        print(f"Generating {num_metal} metal samples...")
        for i in range(num_metal):
            if include_clutter and np.random.rand() < 0.3:
                heatmap, label = self.generate_cluttered_metal_sample()
            else:
                heatmap, label = self.generate_metal_sample()
            X.append(heatmap)
            y.append(label)
        
        print(f"Generating {num_nonmetal} non-metal samples...")
        for i in range(num_nonmetal):
            heatmap, label = self.generate_nonmetal_sample()
            X.append(heatmap)
            y.append(label)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Shuffle
        indices = np.random.permutation(num_samples)
        X = X[indices]
        y = y[indices]
        
        print(f"Dataset generated: {X.shape}, Labels: {y.shape}")
        print(f"Metal samples: {np.sum(y)}, Non-metal samples: {np.sum(1-y)}")
        
        return X, y
    
    def save_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        filepath: str
    ):
        """Save dataset to disk"""
        np.savez_compressed(filepath, X=X, y=y)
        print(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset from disk"""
        data = np.load(filepath)
        return data['X'], data['y']


def augment_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """
    Apply data augmentation to radar heatmap
    
    Args:
        heatmap: Input heatmap
        
    Returns:
        Augmented heatmap
    """
    augmented = heatmap.copy()
    
    # Random flip (horizontal - Doppler axis)
    if np.random.rand() < 0.5:
        augmented = np.flip(augmented, axis=0)
    
    # Random noise
    if np.random.rand() < 0.3:
        noise = np.random.normal(0, 0.02, augmented.shape)
        augmented = augmented + noise
        augmented = np.clip(augmented, 0, 1)
    
    # Random brightness
    if np.random.rand() < 0.3:
        factor = np.random.uniform(0.8, 1.2)
        augmented = augmented * factor
        augmented = np.clip(augmented, 0, 1)
    
    return augmented


def create_train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into training and validation sets
    
    Args:
        X: Features
        y: Labels
        val_ratio: Validation set ratio
        random_state: Random seed
        
    Returns:
        X_train, X_val, y_train, y_val
    """
    np.random.seed(random_state)
    
    num_samples = len(X)
    num_val = int(num_samples * val_ratio)
    
    indices = np.random.permutation(num_samples)
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]
    
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    
    return X_train, X_val, y_train, y_val
