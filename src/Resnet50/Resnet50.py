import torch
import torch.nn as nn

class HomomorphicMultiply(nn.Module):
    def __init__(self, C1):
        super(HomomorphicMultiply, self).__init__()
        self.C1 = C1

    def forward(self, encrypted1, encrypted2, scale=1):
        if encrypted1.size() != encrypted2.size():
            raise ValueError("Input tensors must have the same shape for homomorphic multiplication.")

        encrypted_product = encrypted1 * encrypted2 * self.C1 * scale
        return encrypted_product

class HomomorphicAdd(nn.Module):
    def __init__(self, scaling_factor=1):
        super(HomomorphicAdd, self).__init__()
        self.scaling_factor = scaling_factor

    def forward(self, encrypted1, encrypted2):
        if encrypted1.size() != encrypted2.size():
            raise ValueError("Input tensors must have the same shape for homomorphic addition.")
        
        encrypted_sum = (encrypted1 + encrypted2) * self.scaling_factor
        return encrypted_sum


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
        noise_level = torch.std(encrypted).item()
        
        if noise_level > noise_threshold:
            print("Noise level is high, performing bootstrapping...")
            encrypted = self.reencrypt(encrypted)
        
        return encrypted

    def reencrypt(self, encrypted):
        encrypted = encrypted + torch.randn_like(encrypted) * 0.01
        return encrypted


class BTNK1(nn.Module):
    def __init__(self, C, W, C1, S):
        super(BTNK1, self).__init__()
        
        self.conv1 = nn.Conv2d(C, C1, kernel_size=1, stride=S)
        self.bn1 = nn.BatchNorm2d(C1)
        
        self.conv2 = nn.Conv2d(C1, C1, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(C1)
        
        self.conv3 = nn.Conv2d(C1, C1 * 4, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(C1 * 4)
        
        self.match_identity = nn.Conv2d(C, C1 * 4, kernel_size=1, stride=S)

        self.homomorphic_multiply = HomomorphicMultiply(C1 * 4)
        self.homomorphic_add = HomomorphicAdd()
        self.homomorphic_bootstrap = HomomorphicBootstrapping()

    def forward(self, x):
        identity = self.match_identity(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.homomorphic_multiply(out, identity, scale=1.0)
        out = self.homomorphic_add(out, identity)
        out = self.homomorphic_bootstrap(out)
        out = torch.relu(out)
        return out

class BTNK2(nn.Module):
    def __init__(self, C, W):
        super(BTNK2, self).__init__()
        self.conv1 = nn.Conv2d(C, C // 4, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(C // 4)
        self.conv2 = nn.Conv2d(C // 4, C // 4, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(C // 4)
        self.conv3 = nn.Conv2d(C // 4, C, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(C)
        self.match_identity = nn.Conv2d(C, C, kernel_size=1, stride=1)
        self.homomorphic_multiply = HomomorphicMultiply(C)
        self.homomorphic_add = HomomorphicAdd()
        self.homomorphic_bootstrap = HomomorphicBootstrapping()

    def forward(self, x):
        identity = self.match_identity(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out = self.homomorphic_multiply(out, identity, scale=1.0)
        out = self.homomorphic_add(out, identity)
        out = self.homomorphic_bootstrap(out)
        out = torch.relu(out)
        return out

class Stage0(nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class Stage1(nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.btnk1 = BTNK1(64, 56, 64, 1)
        self.btnk21 = BTNK2(256, 56)
        self.btnk22 = BTNK2(256, 56)

    def forward(self, x):
        x = self.btnk1(x)
        x = self.btnk21(x)
        x = self.btnk22(x)
        return x
    
class Stage2(nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()
        self.btnk1 = BTNK1(256, 56, 128, 2)
        self.btnk21 = BTNK2(512, 28)
        self.btnk22 = BTNK2(512, 28)
        self.btnk23 = BTNK2(512, 28)

    def forward(self, x):
        x = self.btnk1(x)
        x = self.btnk21(x)
        x = self.btnk22(x)
        x = self.btnk23(x)
        return x
    
class Stage3(nn.Module):
    def __init__(self):
        super(Stage3, self).__init__()
        self.btnk1 = BTNK1(512,28,256,2)
        self.btnk21 = BTNK2(1024, 14)
        self.btnk22 = BTNK2(1024, 14)
        self.btnk23 = BTNK2(1024, 14)
        self.btnk24 = BTNK2(1024, 14)
        self.btnk25 = BTNK2(1024, 14)

    def forward(self, x):
        x = self.btnk1(x)
        x = self.btnk21(x)
        x = self.btnk22(x)
        x = self.btnk23(x)
        x = self.btnk24(x)
        x = self.btnk25(x)
        return x
    
class Stage4(nn.Module):
    def __init__(self):
        super(Stage4, self).__init__()
        self.btnk1 = BTNK1(1024,14,512,2)
        self.btnk21 = BTNK2(2048, 7)
        self.btnk22 = BTNK2(2048, 7)

    def forward(self, x):
        x = self.btnk1(x)
        x = self.btnk21(x)
        x = self.btnk22(x)
        return x


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.stage0 = Stage0()
        self.stage1 = Stage1()
        self.stage2 = Stage2()
        self.stage3 = Stage3()
        self.stage4 = Stage4()

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x



model = ResNet50()


x = torch.randn(1, 3, 224, 224)

torch.onnx.export(model, x, "resnet50.onnx", export_params=True, opset_version=11,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 
                                'output': {0: 'batch_size'}})

with torch.no_grad():
    output = model(x)
    print("Output shape:", output.shape)

print("ResNet50 has been exported as onnx file.")

