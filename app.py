from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import os

from actual_model import MILModel  
from create_bags import creating_bags
from load_dataset import MILDataset

# FastAPI app setup
app = FastAPI()

# CORS (optional, useful if frontend is separate)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Static & templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MILModel().to(device)
model.load_state_dict(torch.load("trained_MIL.pth", map_location=device))
model.eval()

# Helper to prepare test data
def prepare_randomized_test_loader():
    mnist_test = datasets.MNIST(root='./data', train=False, download=True)
    
    # Filter for 0s and 7s
    filtered = [(img, label) for img, label in zip(mnist_test.data, mnist_test.targets) if label in [0, 7]]
    images, labels = zip(*filtered)
    images = torch.stack(images).unsqueeze(1).float() / 255.0
    labels = torch.tensor(labels)

    # Shuffle
    perm = torch.randperm(len(images))
    images = images[perm]
    labels = labels[perm]

    # Create bags
    bags, targets = creating_bags(images, labels, num_bags=100)
    test_dataset = MILDataset(bags, targets)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    return test_loader

# Main route
@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    predictions = []
    actuals = []
    actual_counts = []
    predicted_counts = []
    percentage_pairs = []

    test_loader = prepare_randomized_test_loader()
    bags_processed = 0

    with torch.no_grad():
        for bags, targets in test_loader:
            for bag, target in zip(bags, targets):
                pred = model(bag.unsqueeze(0).to(device)).item()
                actual = target.item()

                predictions.append(pred)
                actuals.append(actual)
                percentage_pairs.append((round(pred * 100, 2), round(actual * 100, 2)))

                num_instances = len(bag)
                actual_zeros = int(round(actual * num_instances))
                predicted_zeros = int(round(pred * num_instances))

                actual_counts.append(actual_zeros)
                predicted_counts.append(predicted_zeros)


    mae = sum(abs(p - a) for p, a in zip(predictions, actuals)) / len(actuals)

    plot_file = "predicted_vs_actual.png"
    plt.figure(figsize=(10, 5))
    plt.plot([a * 100 for a in actuals], label="Targets (%)", marker='o')
    plt.plot([p * 100 for p in predictions], label="Predictions (%)", marker='x')
    plt.xlabel("Bag Index")
    plt.ylabel("Percentage of 0's")
    plt.title("Predicted vs Actual (First 5 Bags)")
    plt.legend()
    os.makedirs("static", exist_ok=True)
    plt.savefig(f"static/{plot_file}")
    plt.close()

    bag_data = list(zip(
        range(1, len(actuals) + 1),
        [round(a * 100, 2) for a in actuals],
        [round(p * 100, 2) for p in predictions],
        actual_counts,
        predicted_counts
    ))

    return templates.TemplateResponse("index.html", {
        "request": request,
        "mae": round(mae, 4),
        "samples": percentage_pairs,
        "plot_file": plot_file,
        "bag_data": bag_data
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
