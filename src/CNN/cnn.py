import torch
import torch.nn as nn

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

        encrypted_product = self.homomorphic_multiply(x, self.weights.T)
        
        encrypted_sum = self.homomorphic_add(encrypted_product, self.bias)
        
        encrypted_output = self.bootstrapping(encrypted_sum)
        
        return encrypted_output

class CNN(nn.Module):
    def __init__(self, num_classes=10, homomorphic_multiply=None, homomorphic_add=None, homomorphic_bootstrapping=None):
        super(CNN, self).__init__()

        self.homomorphic_multiply = homomorphic_multiply
        self.homomorphic_add = homomorphic_add
        self.homomorphic_bootstrapping = homomorphic_bootstrapping

        # Convolution layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Fully connected layers (replaced nn.Linear with HomomorphicFC)
        # Correct the input size for fc1 based on the output shape of the convolution layers
        self.fc1 = HomomorphicFC(128 * 3 * 3, 1024, scaling_factor=1, C1=1)  # 128 channels, 3x3 spatial size
        self.fc2 = HomomorphicFC(1024, 512, scaling_factor=1, C1=1)
        self.fc3 = HomomorphicFC(512, num_classes, scaling_factor=1, C1=1)

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)

        x = self.fc3(x)
        return x

homomorphic_multiply = HomomorphicMultiply(C1=1.0)
homomorphic_add = HomomorphicAdd(scaling_factor=1)
homomorphic_bootstrapping = HomomorphicBootstrapping()

model = CNN(num_classes=10,  # Number of output classes (e.g., 10 for MNIST)
            homomorphic_multiply=homomorphic_multiply,
            homomorphic_add=homomorphic_add,
            homomorphic_bootstrapping=homomorphic_bootstrapping)

model.eval()

# Create a random input tensor (batch_size=1, channels=1, height=28, width=28)
x = torch.randn(1, 1, 28, 28)

# Export the model to ONNX
torch.onnx.export(model, x, "CNN_task_graph.onnx", export_params=True, opset_version=13,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 
                                'output': {0: 'batch_size'}})

with torch.no_grad():
    output = model(x)
    print("Output shape:", output.shape)

print("CNN with homomorphic operations has been exported as ONNX file.")