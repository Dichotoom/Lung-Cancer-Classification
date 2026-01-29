import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

class LungCancerClassifier(nn.Module):
    """
    Modular classifier for Lung Cancer Histopathology.
    Supports ResNet50 and EfficientNet-B0 backbones.
    """
    
    def __init__(self, backbone_name: str = 'resnet50', num_classes: int = 3, pretrained: bool = True, dropout: float = 0.3):
        super(LungCancerClassifier, self).__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        
        if backbone_name == 'resnet50':
            # Initialize ResNet50
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            
            # Extract features before the final FC layer
            in_features = self.backbone.fc.in_features
            
            # We keep everything except the final fc layer
            self.backbone.fc = nn.Identity()
            
            # Custom Classifier Head
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes)
            )
            
        elif backbone_name == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            
            # EfficientNet structure: features, avgpool, classifier
            in_features = self.backbone.classifier[1].in_features
            
            # Keep only the features and avgpool
            self.backbone.classifier = nn.Identity()
            
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes)
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose 'resnet50' or 'efficientnet_b0'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def get_last_conv_layer(self) -> nn.Module:
        """
        Returns the last convolutional layer for Grad-CAM.
        """
        if self.backbone_name == 'resnet50':
            return self.backbone.layer4[-1].conv3
        elif self.backbone_name == 'efficientnet_b0':
            return self.backbone.features[-1][0]
        else:
            raise NotImplementedError()

def get_model(backbone_name: str = 'resnet50', num_classes: int = 3, pretrained: bool = True, dropout: float = 0.3) -> LungCancerClassifier:
    """
    Factory function to initialize the model.
    """
    return LungCancerClassifier(backbone_name=backbone_name, num_classes=num_classes, pretrained=pretrained, dropout=dropout)
