from feature_extractor import FeatureExtractor
from dist_pooling import DistributionPoolingFilter
from regression import Regressor
import torch.nn as nn

class MILModel(nn.Module):
    def __init__(self):
        super(MILModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.pooling = DistributionPoolingFilter()
        self.regressor = Regressor()
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pooling(x)
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.regressor(x)
        return x