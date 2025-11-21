"""
Visualization Module
Handles all plotting and visualization functions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
from matplotlib.colors import LinearSegmentedColormap


# Set default style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_range_profile(
    range_profile: np.ndarray,
    max_range: float,
    title: str = "Range Profile",
    save_path: Optional[str] = None
):
    """
    Plot 1D range profile
    
    Args:
        range_profile: 1D range data
        max_range: Maximum range in meters
        title: Plot title
        save_path: Path to save figure
    """
    range_bins = np.linspace(0, max_range, len(range_profile))
    
    plt.figure(figsize=(10, 4))
    plt.plot(range_bins, range_profile, linewidth=2)
    plt.xlabel('Range (m)', fontsize=12)
    plt.ylabel('Magnitude', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_doppler_profile(
    doppler_profile: np.ndarray,
    max_velocity: float,
    title: str = "Doppler Profile",
    save_path: Optional[str] = None
):
    """
    Plot 1D Doppler profile
    
    Args:
        doppler_profile: 1D Doppler data
        max_velocity: Maximum velocity in m/s
        title: Plot title
        save_path: Path to save figure
    """
    velocity_bins = np.linspace(-max_velocity, max_velocity, len(doppler_profile))
    
    plt.figure(figsize=(10, 4))
    plt.plot(velocity_bins, doppler_profile, linewidth=2, color='orange')
    plt.xlabel('Velocity (m/s)', fontsize=12)
    plt.ylabel('Magnitude', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_range_doppler_heatmap(
    rd_map: np.ndarray,
    max_range: float,
    max_velocity: float,
    title: str = "Range-Doppler Heatmap",
    cmap: str = 'jet',
    log_scale: bool = True,
    save_path: Optional[str] = None
):
    """
    Plot 2D range-Doppler heatmap
    
    Args:
        rd_map: 2D range-Doppler magnitude map
        max_range: Maximum range in meters
        max_velocity: Maximum velocity in m/s
        title: Plot title
        cmap: Colormap
        log_scale: Use logarithmic scale
        save_path: Path to save figure
    """
    if log_scale:
        data = 20 * np.log10(rd_map + 1e-10)
        cbar_label = 'Magnitude (dB)'
    else:
        data = rd_map
        cbar_label = 'Magnitude'
    
    plt.figure(figsize=(10, 6))
    
    extent = [0, max_range, -max_velocity, max_velocity]
    
    im = plt.imshow(
        data,
        aspect='auto',
        cmap=cmap,
        extent=extent,
        origin='lower',
        interpolation='bilinear'
    )
    
    plt.colorbar(im, label=cbar_label)
    plt.xlabel('Range (m)', fontsize=12)
    plt.ylabel('Velocity (m/s)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axhline(y=0, color='white', linestyle='--', alpha=0.5, linewidth=1)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_multiple_scenarios(
    scenarios: List[Tuple[np.ndarray, str]],
    max_range: float,
    max_velocity: float,
    save_path: Optional[str] = None
):
    """
    Plot multiple scenarios in a grid
    
    Args:
        scenarios: List of (rd_map, title) tuples
        max_range: Maximum range in meters
        max_velocity: Maximum velocity in m/s
        save_path: Path to save figure
    """
    num_scenarios = len(scenarios)
    cols = min(3, num_scenarios)
    rows = (num_scenarios + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    if num_scenarios == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    extent = [0, max_range, -max_velocity, max_velocity]
    
    for idx, (rd_map, title) in enumerate(scenarios):
        data = 20 * np.log10(rd_map + 1e-10)
        
        im = axes[idx].imshow(
            data,
            aspect='auto',
            cmap='jet',
            extent=extent,
            origin='lower',
            interpolation='bilinear'
        )
        
        axes[idx].set_xlabel('Range (m)')
        axes[idx].set_ylabel('Velocity (m/s)')
        axes[idx].set_title(title, fontweight='bold')
        axes[idx].axhline(y=0, color='white', linestyle='--', alpha=0.5, linewidth=1)
        fig.colorbar(im, ax=axes[idx], label='Magnitude (dB)')
    
    # Hide extra subplots
    for idx in range(num_scenarios, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str] = ['Non-Metal', 'Metal'],
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        classes: Class names
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Count'}
    )
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None
):
    """
    Plot training history (loss and accuracy)
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in history:
        ax2.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sample_predictions(
    X_samples: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_samples: int = 6,
    save_path: Optional[str] = None
):
    """
    Plot sample predictions
    
    Args:
        X_samples: Input samples
        y_true: True labels
        y_pred: Predicted labels
        num_samples: Number of samples to display
        save_path: Path to save figure
    """
    num_samples = min(num_samples, len(X_samples))
    cols = 3
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    class_names = ['Non-Metal', 'Metal']
    
    for idx in range(num_samples):
        im = axes[idx].imshow(X_samples[idx], cmap='jet', aspect='auto')
        
        true_label = class_names[int(y_true[idx])]
        pred_label = class_names[int(y_pred[idx])]
        
        color = 'green' if y_true[idx] == y_pred[idx] else 'red'
        
        axes[idx].set_title(
            f'True: {true_label}\nPred: {pred_label}',
            color=color,
            fontweight='bold'
        )
        axes[idx].axis('off')
    
    # Hide extra subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    save_path: Optional[str] = None
):
    """
    Plot ROC curve
    
    Args:
        fpr: False positive rate
        tpr: True positive rate
        auc_score: AUC score
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 6))
    
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_background_subtraction(
    original: np.ndarray,
    background: np.ndarray,
    foreground: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Visualize background subtraction process
    
    Args:
        original: Original frame
        background: Background model
        foreground: Foreground (subtracted)
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    im1 = axes[0].imshow(original, cmap='jet', aspect='auto')
    axes[0].set_title('Original Frame', fontweight='bold')
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(background, cmap='jet', aspect='auto')
    axes[1].set_title('Background Model', fontweight='bold')
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(foreground, cmap='hot', aspect='auto')
    axes[2].set_title('Foreground (Targets)', fontweight='bold')
    axes[2].axis('off')
    fig.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_noise_filtering(
    noisy: np.ndarray,
    filtered: np.ndarray,
    filter_type: str = 'median',
    save_path: Optional[str] = None
):
    """
    Visualize noise filtering
    
    Args:
        noisy: Noisy signal
        filtered: Filtered signal
        filter_type: Filter type name
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    im1 = axes[0].imshow(noisy, cmap='jet', aspect='auto')
    axes[0].set_title('Noisy Signal', fontweight='bold')
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(filtered, cmap='jet', aspect='auto')
    axes[1].set_title(f'Filtered Signal ({filter_type})', fontweight='bold')
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
