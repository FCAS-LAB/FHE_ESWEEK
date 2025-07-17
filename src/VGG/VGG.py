import torch
import torch.nn as nn
import torch.nn.functional as F

class HomomorphicAdd(nn.Module):
    def __init__(self, scaling_factor=1):
        super(HomomorphicAdd, self).__init__()
        self.scaling_factor = scaling_factor

    def forward(self, encrypted1, encrypted2):
        if encrypted1.size() != encrypted2.size():
            encrypted2 = encrypted2.unsqueeze(0).expand_as(encrypted1)
        encrypted_sum = (encrypted1 + encrypted2) * self.scaling_factor
        return encrypted_sum

class HomomorphicMultiply(nn.Module):
    def __init__(self, C1):
        super(HomomorphicMultiply, self).__init__()
        self.C1 = C1

    def forward(self, encrypted1, encrypted2, scale=1):
        if encrypted1.size(1) != encrypted2.size(0):
            raise ValueError("Input tensors must have compatible shapes for matrix multiplication.")
        
        encrypted_product = torch.matmul(encrypted1, encrypted2) * self.C1 * scale
        return encrypted_product

class HomomorphicBootstrapping(nn.Module):
    def __init__(self, homomorphic_add=None):
        super(HomomorphicBootstrapping, self).__init__()
        if homomorphic_add is None:
            self.homomorphic_add = lambda x, y: x + y
        else:
            self.homomorphic_add = homomorphic_add

    def forward(self, encrypted, noise_threshold=1e-3):
        return self.bootstrap(encrypted, noise_threshold)

    def bootstrap(self, encrypted, noise_threshold=1e-3):
        noise_level = torch.std(encrypted)
        if noise_level > noise_threshold:
            print("Noise level is high, performing bootstrapping...")
            encrypted = self.reencrypt(encrypted)
        return encrypted

    def reencrypt(self, encrypted):
        encrypted = encrypted + torch.randn_like(encrypted) * 0.01
        return encrypted

class HomomorphicFC(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor=1, C1=1):
        super(HomomorphicFC, self).__init__()
        self.homomorphic_multiply = HomomorphicMultiply(C1)
        self.homomorphic_add = HomomorphicAdd(scaling_factor)
        self.bootstrapping = HomomorphicBootstrapping(self.homomorphic_add)

        self.weights = torch.randn(out_features, in_features)
        self.bias = torch.randn(out_features)

    def forward(self, x):
        # Perform the homomorphic multiplication (equivalent to linear transformation)
        encrypted_product = self.homomorphic_multiply(x, self.weights.T) 
        
        # Add bias using homomorphic addition
        encrypted_sum = self.homomorphic_add(encrypted_product, self.bias)
        
        # Bootstrapping to handle noise
        encrypted_output = self.bootstrapping(encrypted_sum)
        
        return encrypted_output


class VGGHomomorphic(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGGHomomorphic, self).__init__()
        self.features = nn.Sequential(
            # Convolution layers (Layer 1-2, output: 224x224x64)
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Convolution layers (Layer 3-4, output: 112x112x128)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Convolution layers (Layer 5-7, output: 56x56x256)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Convolution layers (Layer 8-10, output: 28x28x512)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Convolution layers (Layer 11-13, output: 14x14x512)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        input_features = 512 * 14 * 14  

        # Define the classifier with the correct input features
        self.classifier = nn.Sequential(
            HomomorphicFC(512 * 7 * 7, 4096, scaling_factor=1, C1=1),  # First FC Layer
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            HomomorphicFC(4096, 4096, scaling_factor=1, C1=1),  # Second FC Layer
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            HomomorphicFC(4096, num_classes, scaling_factor=1, C1=1)  # Output Layer
        )

    def forward(self, x):
        x = self.features(x)  # Apply convolution layers

        # Flatten the feature maps into a 2D tensor: (batch_size, input_features)
        x = x.view(x.size(0), -1)  # Flatten the tensor to (batch_size, input_features)

        # Now pass the flattened tensor through the classifier (which includes HomomorphicFCLayer)
        x = self.classifier(x)
        return x

# Testing with a dummy input
model = VGGHomomorphic(num_classes=1000)

# (batch_size=1, channels=3, height=224, width=224)
x = torch.randn(1, 3, 224, 224)
output = model(x)
# Export the model to ONNX
torch.onnx.export(model, x, "VGG_task_graph.onnx", export_params=True, opset_version=13,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 
                                'output': {0: 'batch_size'}})

with torch.no_grad():
    output = model(x)
    print("Output shape:", output.shape)
print("vgg with homomorphic operations has been exported as onnx file.")
