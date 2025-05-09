import torch
import torch.nn as nn
from residual_block import Block


# Define ResNet-50 
class ResNet50(nn.Module):
    def __init__(self, num_classes=102):
        super(ResNet50, self).__init__()
        self.resnet_model = nn.Sequential(

            # Initial convolutional layer
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            #1: 3 Blocks
            Block(64, 64, 256, stride=1),
            Block(256, 64, 256, stride=1),
            Block(256, 64, 256, stride=1),

            # 2: 4 Blocks
            Block(256, 128, 512, stride=2),
            Block(512, 128, 512, stride=1),
            Block(512, 128, 512, stride=1),
            Block(512, 128, 512, stride=1),

            #3: 6 Blocks
            Block(512, 256, 1024, stride=2),
            Block(1024, 256, 1024, stride=1),
            Block(1024, 256, 1024, stride=1),
            Block(1024, 256, 1024, stride=1),
            Block(1024, 256, 1024, stride=1),
            Block(1024, 256, 1024, stride=1),

            #4: 3 Blocks
            Block(1024, 512, 2048, stride=2),
            Block(2048, 512, 2048, stride=1),
            Block(2048, 512, 2048, stride=1),

            # Average Pooling and FC Layer
            nn.AdaptiveAvgPool2d((1, 1)), # Reduce each channel to target size(1,1)
            nn.Flatten(),
            nn.Linear(2048, 102)    # 80 classes for COCO dataset
            
        )

    # Forward pass
    def forward(self, x):
        return self.resnet_model(x)


