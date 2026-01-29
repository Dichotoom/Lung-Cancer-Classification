#!/usr/bin/env python3
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add project root to path
root = Path(__file__).parent.parent
sys.path.append(str(root))

from src.utils.config_loader import Config
from src.utils.paths import PathManager
from src.utils.logging_utils import setup_logger
from src.utils.checkpoint import load_checkpoint
from src.data.data_loader import get_data_loaders
from src.models.classifier import get_model
from src.evaluation.explainability import GradCAM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    args = parser.parse_args()
    
    config = Config(args.config)
    paths = PathManager(config)
    logger = setup_logger('Visualize', paths.get_log_path('visualization.log'))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data
    _, val_loader = get_data_loaders(
        data_dir=paths.get_data_dir(),
        class_names=config.get('data.class_names'),
        img_size=config.get('data.img_size'),
        mean=config.get('normalization.mean'),
        std=config.get('normalization.std'),
        batch_size=1, # Single image
        val_split=config.get('data.val_split'),
        num_workers=config.get('data.num_workers'),
        seed=config.get('project.seed')
    )
    
    # Model
    model = get_model(
        backbone_name=config.get('model.backbone'),
        num_classes=config.get('model.num_classes'),
        pretrained=config.get('model.pretrained')
    ).to(device)
    
    # Load
    checkpoint_path = paths.get_checkpoint_path('best_model.pth')
    try:
        load_checkpoint(checkpoint_path, model)
    except FileNotFoundError:
        logger.warning(f"Checkpoint not found. Using untrained model for demo.")

    model.eval()
    
    # Get Sample
    img_tensor, label_idx = next(iter(val_loader))
    img_tensor = img_tensor.to(device)
    
    logger.info(f"Generating Grad-CAM for class: {config.get('data.class_names')[label_idx.item()]}")
    
    # Grad-CAM
    target_layer = model.get_last_conv_layer()
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam.generate_heatmap(img_tensor)
    
    # Denormalize
    mean = np.array(config.get('normalization.mean'))
    std = np.array(config.get('normalization.std'))
    img_np = img_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)
    img_uint8 = (img_np * 255).astype(np.uint8)
    
    overlay = grad_cam.overlay_heatmap(img_uint8, heatmap)
    
    # Save
    save_path = paths.get_visualization_path('gradcam_demo.png')
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title(f"Original: {config.get('data.class_names')[label_idx.item()]}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Grad-CAM Overlay")
    plt.axis('off')
    
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"Saved Grad-CAM to {save_path}")

if __name__ == "__main__":
    main()
