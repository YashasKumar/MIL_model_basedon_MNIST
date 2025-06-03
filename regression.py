#A simple linear regression should do the trick for the final predictioin
import torch.nn as nn

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.regressors = nn.Sequential(
            nn.Linear(2048,32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        x = self.regressors(x)
        return x