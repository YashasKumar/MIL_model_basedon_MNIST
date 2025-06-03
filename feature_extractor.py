#In the model they used ResNet pretrained models to extract the features, 
#but since I am working on a simpler dataset, I will build a custom smaller CNN network for extracting features
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.feature_extraction = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size = 3, padding = 1),
        # nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.1),
        nn.Conv2d(16, 32, kernel_size = 3, padding = 1),
        # nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        # nn.Dropout(0.1),
        nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.1),
        nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
        )
    
    def forward(self, x):
        batch_size, num_instances, c, h, w = x.shape
        x = x.view(batch_size*num_instances, c, h, w)
        x = self.feature_extraction(x)
        return x.view(batch_size, num_instances, -1)