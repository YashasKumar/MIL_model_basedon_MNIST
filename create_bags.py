import torch
import numpy as np

def creating_bags(images, labels, num_bags, bag_size = 100):
    bags, targets = [], []
    
    for _ in range(num_bags):
        #Assign a random % of 0s to be in a bag
        x = np.random.uniform(0, 1)
        num_zeros = int(bag_size*x)
        num_seven = bag_size-num_zeros
        
        zero_indicies = torch.where(labels == 0)[0]
        seven_indices = torch.where(labels == 7)[0]
        
        if len(zero_indicies)<num_zeros or len(seven_indices)<num_seven:
            break
        
        zero_samples = zero_indicies[torch.randperm(len(zero_indicies))[:num_zeros]]
        seven_samples = seven_indices[torch.randperm(len(seven_indices))[:num_seven]]
        
        selected_final_samples = torch.cat([zero_samples, seven_samples])
        bags.append(images[selected_final_samples])
        
        targets.append(torch.tensor([x], dtype=torch.float32))

        #Filtering out the samples that were already used in one bad, so 2 bags dont have the exact same sample
        mask = torch.ones(len(images), dtype=bool)
        mask[selected_final_samples] = False
        images, labels = images[mask], labels[mask]
    
    return torch.stack(bags), torch.stack(targets)
        