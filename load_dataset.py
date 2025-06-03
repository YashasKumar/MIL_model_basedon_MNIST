from torch.utils.data import Dataset

#Defining the dataset
class MILDataset(Dataset):
    def __init__(self, bags, targets):
        self.bags = bags
        self.targets = targets
    
    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, i):
        return self.bags[i], self.targets[i]