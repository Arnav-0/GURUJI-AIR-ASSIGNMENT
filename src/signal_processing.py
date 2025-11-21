"""
Signal Processing Module
Handles radar signal generation, FFT transformations, and preprocessing
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fft2, fftshift
from typing import Tuple, Dict, Optional


class RadarSimulator:
    """
    Simulates FMCW radar signals with range and Doppler information
    """
    
    def __init__(
        self,
        fc: float = 77e9,  # Center frequency (77 GHz)
        bandwidth: float = 4e9,  # Bandwidth (4 GHz)
        sweep_time: float = 100e-6,  # Chirp duration (100 μs)
        num_chirps: int = 128,  # Number of chirps
        num_samples: int = 256,  # Samples per chirp
        c: float = 3e8  # Speed of light
    ):
        """
        Initialize radar parameters
        
        Args:
            fc: Center frequency in Hz
            bandwidth: Sweep bandwidth in Hz
            sweep_time: Chirp duration in seconds
            num_chirps: Number of chirps in frame
            num_samples: ADC samples per chirp
            c: Speed of light in m/s
        """
        self.fc = fc
        self.bandwidth = bandwidth
        self.sweep_time = sweep_time
        self.num_chirps = num_chirps
        self.num_samples = num_samples
        self.c = c
        
        # Derived parameters
        self.slope = bandwidth / sweep_time
        self.sample_rate = num_samples / sweep_time
        self.range_resolution = c / (2 * bandwidth)
        self.max_range = (c * self.sample_rate) / (2 * self.slope)
        
        # Doppler parameters
        self.frame_time = num_chirps * sweep_time
        self.doppler_resolution = c / (2 * fc * self.frame_time)
        self.max_velocity = (c * num_chirps) / (4 * fc * sweep_time)
        
    def generate_target_signal(
        self,
        ranges: np.ndarray,
        velocities: np.ndarray,
        rcs: np.ndarray,
        noise_power: float = 0.01
    ) -> np.ndarray:
        """
        Generate radar signal for multiple targets
        
        Args:
            ranges: Array of target ranges in meters
            velocities: Array of target velocities in m/s
            rcs: Radar cross-section values (amplitude scaling)
            noise_power: Noise power level
            
        Returns:
            Complex radar data cube (num_chirps x num_samples)
        """
        # Time vectors
        t_fast = np.linspace(0, self.sweep_time, self.num_samples)
        t_slow = np.arange(self.num_chirps) * self.sweep_time
        
        # Initialize signal matrix
        signal_matrix = np.zeros((self.num_chirps, self.num_samples), dtype=complex)
        
        # Add each target
        for r, v, amp in zip(ranges, velocities, rcs):
            # Beat frequency (range encoding)
            f_beat = (2 * self.slope * r) / self.c
            
            # Doppler frequency (velocity encoding)
            f_doppler = (2 * v * self.fc) / self.c
            
            # Generate signal for this target
            for chirp_idx in range(self.num_chirps):
                phase_doppler = 2 * np.pi * f_doppler * t_slow[chirp_idx]
                target_signal = amp * np.exp(
                    1j * 2 * np.pi * f_beat * t_fast + 1j * phase_doppler
                )
                signal_matrix[chirp_idx, :] += target_signal
        
        # Add complex Gaussian noise
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(self.num_chirps, self.num_samples) +
            1j * np.random.randn(self.num_chirps, self.num_samples)
        )
        signal_matrix += noise
        
        return signal_matrix
    
    def generate_empty_room(self, noise_power: float = 0.01) -> np.ndarray:
        """Generate signal for empty room scenario"""
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(self.num_chirps, self.num_samples) +
            1j * np.random.randn(self.num_chirps, self.num_samples)
        )
        return noise
    
    def generate_metal_object(
        self,
        range_m: float = 2.0,
        velocity: float = 0.5,
        rcs: float = 1.0,
        noise_power: float = 0.01
    ) -> np.ndarray:
        """Generate signal for metal object scenario"""
        return self.generate_target_signal(
            ranges=np.array([range_m]),
            velocities=np.array([velocity]),
            rcs=np.array([rcs]),
            noise_power=noise_power
        )
    
    def generate_clutter_scenario(
        self,
        num_targets: int = 5,
        noise_power: float = 0.02
    ) -> np.ndarray:
        """Generate cluttered scene with multiple objects"""
        ranges = np.random.uniform(0.5, 5.0, num_targets)
        velocities = np.random.uniform(-2.0, 2.0, num_targets)
        rcs = np.random.uniform(0.3, 1.2, num_targets)
        
        return self.generate_target_signal(ranges, velocities, rcs, noise_power)
    
    def range_fft(self, signal_matrix: np.ndarray) -> np.ndarray:
        """
        Apply FFT along range dimension (fast-time)
        
        Args:
            signal_matrix: Raw radar data (num_chirps x num_samples)
            
        Returns:
            Range FFT output
        """
        return fft(signal_matrix, axis=1)
    
    def doppler_fft(self, range_fft_data: np.ndarray) -> np.ndarray:
        """
        Apply FFT along Doppler dimension (slow-time)
        
        Args:
            range_fft_data: Range FFT output
            
        Returns:
            Range-Doppler map
        """
        doppler_data = fft(range_fft_data, axis=0)
        return fftshift(doppler_data, axes=0)
    
    def generate_range_doppler_map(self, signal_matrix: np.ndarray) -> np.ndarray:
        """
        Generate 2D range-Doppler heatmap
        
        Args:
            signal_matrix: Raw radar signal
            
        Returns:
            2D range-Doppler magnitude map
        """
        range_fft_data = self.range_fft(signal_matrix)
        doppler_data = self.doppler_fft(range_fft_data)
        return np.abs(doppler_data)
    
    def get_range_profile(self, signal_matrix: np.ndarray) -> np.ndarray:
        """Get 1D range profile (averaged across chirps)"""
        range_fft_data = self.range_fft(signal_matrix)
        return np.mean(np.abs(range_fft_data), axis=0)
    
    def get_doppler_profile(self, signal_matrix: np.ndarray) -> np.ndarray:
        """Get 1D Doppler profile (averaged across range bins)"""
        range_fft_data = self.range_fft(signal_matrix)
        doppler_data = self.doppler_fft(range_fft_data)
        return np.mean(np.abs(doppler_data), axis=1)
    
    def apply_windowing(
        self,
        signal_matrix: np.ndarray,
        window_type: str = 'hamming'
    ) -> np.ndarray:
        """
        Apply window function to reduce spectral leakage
        
        Args:
            signal_matrix: Input signal
            window_type: 'hamming', 'hanning', 'blackman', 'kaiser'
            
        Returns:
            Windowed signal
        """
        if window_type == 'hamming':
            window_range = np.hamming(self.num_samples)
            window_doppler = np.hamming(self.num_chirps)
        elif window_type == 'hanning':
            window_range = np.hanning(self.num_samples)
            window_doppler = np.hanning(self.num_chirps)
        elif window_type == 'blackman':
            window_range = np.blackman(self.num_samples)
            window_doppler = np.blackman(self.num_chirps)
        else:
            return signal_matrix
        
        # Apply 2D window
        window_2d = np.outer(window_doppler, window_range)
        return signal_matrix * window_2d
    
    def cfar_detection(
        self,
        range_doppler_map: np.ndarray,
        guard_cells: int = 4,
        training_cells: int = 8,
        pfa: float = 1e-4
    ) -> np.ndarray:
        """
        CFAR (Constant False Alarm Rate) detection
        
        Args:
            range_doppler_map: Input magnitude map
            guard_cells: Number of guard cells
            training_cells: Number of training cells
            pfa: Probability of false alarm
            
        Returns:
            Binary detection map
        """
        threshold_factor = -np.log(pfa)
        detections = np.zeros_like(range_doppler_map)
        
        num_doppler, num_range = range_doppler_map.shape
        
        for i in range(guard_cells + training_cells, num_doppler - guard_cells - training_cells):
            for j in range(guard_cells + training_cells, num_range - guard_cells - training_cells):
                # Extract training region
                region = range_doppler_map[
                    i - guard_cells - training_cells:i + guard_cells + training_cells + 1,
                    j - guard_cells - training_cells:j + guard_cells + training_cells + 1
                ]
                
                # Exclude guard cells and CUT
                mask = np.ones_like(region, dtype=bool)
                mask[
                    training_cells:training_cells + 2 * guard_cells + 1,
                    training_cells:training_cells + 2 * guard_cells + 1
                ] = False
                
                # Calculate threshold
                noise_level = np.mean(region[mask])
                threshold = noise_level * threshold_factor
                
                # Detection
                if range_doppler_map[i, j] > threshold:
                    detections[i, j] = 1
        
        return detections


def apply_background_subtraction(
    current_frame: np.ndarray,
    background: np.ndarray,
    alpha: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply background subtraction with running average
    
    Args:
        current_frame: Current radar frame
        background: Background model
        alpha: Learning rate for background update
        
    Returns:
        Foreground and updated background
    """
    foreground = np.abs(current_frame - background)
    background_updated = alpha * current_frame + (1 - alpha) * background
    return foreground, background_updated


def apply_noise_filtering(
    signal_data: np.ndarray,
    filter_type: str = 'median',
    kernel_size: int = 3
) -> np.ndarray:
    """
    Apply noise filtering to radar data
    
    Args:
        signal_data: Input signal/image
        filter_type: 'median', 'gaussian', 'bilateral'
        kernel_size: Filter kernel size
        
    Returns:
        Filtered signal
    """
    if filter_type == 'median':
        from scipy.ndimage import median_filter
        return median_filter(signal_data, size=kernel_size)
    elif filter_type == 'gaussian':
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(signal_data, sigma=kernel_size/3)
    else:
        return signal_data


def normalize_heatmap(heatmap: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize heatmap for visualization and model input
    
    Args:
        heatmap: Input heatmap
        method: 'minmax', 'zscore', 'log'
        
    Returns:
        Normalized heatmap
    """
    if method == 'minmax':
        min_val, max_val = heatmap.min(), heatmap.max()
        if max_val - min_val > 0:
            return (heatmap - min_val) / (max_val - min_val)
        return heatmap
    elif method == 'zscore':
        mean, std = heatmap.mean(), heatmap.std()
        if std > 0:
            return (heatmap - mean) / std
        return heatmap
    elif method == 'log':
        return np.log10(heatmap + 1e-10)
    return heatmap
