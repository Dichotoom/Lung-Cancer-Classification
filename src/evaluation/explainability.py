import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional

class GradCAM:
    """
    Grad-CAM implementation for visual explanations.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.hooks = [
            target_layer.register_forward_hook(self.save_activation),
            target_layer.register_full_backward_hook(self.save_gradient)
        ]

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generates a Grad-CAM heatmap.
        """
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()
        
        # Pool the gradients across the channels
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the channels by corresponding gradients
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the activations
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        
        # ReLU on top of the heatmap
        heatmap = F.relu(heatmap)
        
        # Normalize the heatmap
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
        
        return heatmap.detach().cpu().numpy()

    def overlay_heatmap(self, img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """
        Overlays heatmap on the original image.
        
        Args:
            img: Original image as uint8 numpy array
            heatmap: Normalized heatmap (0-1 range)
            alpha: Blending factor for heatmap
            
        Returns:
            Blended image as uint8 numpy array
        """
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Convert heatmap to RGB
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Overlay - ensure consistent dtype
        overlayed_img = heatmap_color.astype(float) * alpha + img.astype(float) * (1 - alpha)
        overlayed_img = np.clip(overlayed_img, 0, 255).astype(np.uint8)
        
        return overlayed_img

    def __del__(self):
        for hook in self.hooks:
            hook.remove()
