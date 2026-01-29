import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
from typing import Optional, Tuple
from pathlib import Path
from src.utils.checkpoint import save_checkpoint

class Trainer:
    """
    Trainer class for Lung Cancer classification.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        checkpoint_dir: Path
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("Trainer")
        self.best_acc = 0.0

    def train_epoch(self, limit_batches: Optional[int] = None) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for i, (images, labels) in enumerate(pbar):
            if limit_batches and i >= limit_batches:
                break
                
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
            
        return running_loss / total, 100. * correct / total

    @torch.no_grad()
    def validate(self, limit_batches: Optional[int] = None) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(self.val_loader):
            if limit_batches and i >= limit_batches:
                break
                
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return running_loss / total, 100. * correct / total

    def fit(self, epochs: int, limit_batches: Optional[int] = None):
        self.logger.info(f"Starting training for {epochs} epochs on {self.device}")
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(limit_batches=limit_batches)
            val_loss, val_acc = self.validate(limit_batches=limit_batches)
            
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Save Best Checkpoint
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_acc': self.best_acc,
            }, is_best, self.checkpoint_dir)
            
            if is_best:
                self.logger.info(f"Best model saved with accuracy: {self.best_acc:.2f}%")