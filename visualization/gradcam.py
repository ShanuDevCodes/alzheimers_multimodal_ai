import torch
import torch.nn.functional as F
import numpy as np

class GradCAM3D:
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class=None):
        
        self.model.eval()
        logits, _ = self.model(input_tensor)
        
        if target_class is None:
            score = logits.squeeze()
        else:
            score = logits[0, target_class]
            
        self.model.zero_grad()
        score.backward()

        gradients = self.gradients.detach()
        activations = self.activations.detach()

        weights = torch.mean(gradients, dim=(2, 3, 4), keepdim=True)

        cam = torch.sum(weights * activations, dim=1).squeeze()
        cam = F.relu(cam) # Apply ReLU to keep only positive influences

        cam -= torch.min(cam)
        cam /= torch.max(cam) + 1e-8

        return cam.cpu().numpy()

def overlay_cam_3d(mri_volume, cam_volume):
    
    import scipy.ndimage as ndimage
    
    zoom_factors = (
        mri_volume.shape[0] / cam_volume.shape[0],
        mri_volume.shape[1] / cam_volume.shape[1],
        mri_volume.shape[2] / cam_volume.shape[2]
    )
    cam_resized = ndimage.zoom(cam_volume, zoom_factors, order=1)
    
    return mri_volume, cam_resized
