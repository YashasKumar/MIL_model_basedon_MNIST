import matplotlib.pyplot as plt
import os
import numpy as np

def plot_loss(loss_vals, filename, save_path="results_graphs"):
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Plot loss values
    indices = np.arange(len(loss_vals))
    plt.figure(figsize=(8, 5))
    plt.plot(indices, loss_vals, label="Loss", color="blue")

    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.title("Loss Over Epochs") 
    plt.legend()

    # Save the figure
    save_file = os.path.join(save_path, filename)
    plt.savefig(save_file)
    plt.close()

    print(f"Graph saved at: {save_file}")
