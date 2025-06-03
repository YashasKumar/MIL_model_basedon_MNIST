import matplotlib.pyplot as plt
import os
import numpy as np

def plot_save_targets_distribution(loader, filename, save_path="results_graphs"):
    percentages = []

    # Collect all target percentages
    for _, targets in loader:
        targets = targets.cpu().numpy().flatten()  # Convert to 1D numpy array
        percentages.extend(targets)  # Store percentages of '0's

    percentages = np.array(percentages)
    percentage_of_7s = 1 - percentages  # Since it's a binary problem (0s and 7s)

    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Plot stacked bar chart
    indices = np.arange(len(percentages))
    plt.figure(figsize=(8, 5))
    plt.bar(indices, percentages * 100, label="0s (%)", color="blue")
    plt.bar(indices, percentage_of_7s * 100, bottom=percentages * 100, label="7s (%)", color="red")
    
    plt.xlabel("Sample Index")
    plt.ylabel("Percentage")
    plt.title("Distribution of 0s and 7s in Targets")
    plt.legend()

    # Save the figure
    save_file = os.path.join(save_path, filename)
    plt.savefig(save_file)
    plt.close()

    print(f"Graph saved at: {save_file}")
