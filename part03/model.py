import torch.nn as nn

class BaselineCNN(nn.Module):
    """
    Custom 3 block CNN:
    
    >   Network Structure:
    >       Convolutional Layers: Stack at least 3 convolutional layers with increasing filter sizes (e.g., 32, 64, 128) using ReLU activation.
    >       Pooling Layers: Apply Max-Pooling after convolutional blocks to downsample spatial dimensions.
    >       Dense Layers: Flatten the feature maps and pass them through at least one fully connected hidden layer before the final softmax output layer.
    """

    def __init__(self, num_classes=6, dropout=0.0):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: [3, 128, 128] -> [32, 64, 64]
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2: [32, 64, 64] -> [64, 32, 32]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: [64, 32, 32] -> [128, 16, 16]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Output: [128, 16, 16]
        )

        # Dense layers: y = xW^T + b
        self.classifier = nn.Sequential(
            # x = [128, 16, 16]

            nn.Flatten(),
            # x = [1, 32768]

            nn.Linear(128 * 16 * 16, 256), # FC hidden layer
            # W = [256, 32768]
            # b = [1, 256]
            # y = [1, 32768] @ [32768, 256] + [1, 256] = [1, 256]
            # x = y

            nn.ReLU(inplace=True), # discard negative values. x = max(0, x)

            nn.Dropout(dropout), # regularization knob

            nn.Linear(256, num_classes),
            # x = [1, 256]
            # W = [6, 256]
            # b = [1, 6]
            # y = [1, 256] @ [256, 6] + [1, 6] = [1, 6]
        )

    def forward(self, x):
        # Forward pass through the network.
        x = self.features(x)        # [3, 128, 128] -> [128, 16, 16]
        return self.classifier(x)   # [128, 16, 16] -> [1, 6]
