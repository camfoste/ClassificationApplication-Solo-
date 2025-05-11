import torch
import torch.nn as nn
import torch.nn.functional as F

'''
docstring
          
'''

class Net(nn.Module):
    def __init__(self, num_classes):
        """
        Initializes the convolutional neural network.

        Args:
            num_classes (int): Number of output classes for classification.

        Architecture:
            - Convolutional layers:
                - conv1: 3 input channels to 16 filters (3x3, same padding)
                - conv2: 16 filters to 32 filters (3x3, same padding)
                - conv3: 32 filters to 64 filters (3x3, same padding)
            - Pooling: Max pooling (2x2) applied after each convolutional layer.
            - Fully connected layers:
                - fc1: Flattens and connects to 512 neurons.
                - Dropout (0.2) applied for regularization.
                - fc2: 512 neurons to 256 neurons.
                - fc3: 256 neurons to `num_classes` (output layer).
        """
        
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding='same')
        self.conv2 = nn.Conv2d(16, 32, 3, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding='same')
        self.fc1 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).

        Steps:
            1. Passes input through convolutional layers with ReLU activation and max pooling.
            2. Flattens the feature maps into a 1D vector.
            3. Applies dropout and passes through fully connected layers.
            4. Outputs raw class scores (logits).
        """
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
