import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve, auc
from typing import List

class Visualizer:
    """Utilities for saving confusion matrices and ROC curves."""
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], filename: str = 'confusion_matrix.png') -> Path:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path

    def plot_roc_curves(self, y_true: np.ndarray, y_probs: np.ndarray, class_names: List[str], filename: str = 'roc_curves.png') -> Path:
        plt.figure(figsize=(10, 8))
        for i in range(len(class_names)):
            # One-vs-Rest ROC
            fpr, tpr, _ = roc_curve(y_true == i, y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curves')
        plt.legend(loc='lower right')
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
