import torch
import matplotlib.pyplot as plt
import os

def evaluate_model(model, test_loader, device, test_or_train, save_path="results_graphs/targetVSpredicted.png"):
    model.eval()
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        total_error = 0
        for bags, targets in test_loader:
            bags, targets = bags.to(device), targets.to(device)
            outputs = model(bags)
            error = torch.abs(outputs - targets).mean().item()
            total_error += error

            all_outputs.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    if test_or_train:
        print("MAE during evaluation of training data is: {:.4f}".format(total_error / len(test_loader)))
    else:
        print("MAE during evaluation of testing data is: {:.4f}".format(total_error / len(test_loader)))

    # Create results_graph directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Plot the outputs vs targets
    plt.figure(figsize=(10, 5))
    plt.plot(all_targets, label="Targets", marker='o', linestyle='-')
    plt.plot(all_outputs, label="Outputs", marker='x', linestyle='-')
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.title("Model Predictions vs Targets")
    plt.legend()

    # Save the graph
    plt.savefig(save_path)
    plt.close()
    print(f"Graph saved at {save_path}")
