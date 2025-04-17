import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Import the custom dataloader and dataset implementation from Dataloader.py
from Dataloader import FaceDataset, FaceDataLoader

############################################
# Custom image transformations (no torchvision.transforms)
############################################

class Resize(nn.Module):
    """
    Resize image tensor using bilinear interpolation.
    Expects input of shape (N, C, H, W) or (C, H, W).
    """
    def __init__(self, size):
        super().__init__()
        self.size = size
        
    def forward(self, x):
        # If x is a single image (C,H,W), add batch dimension.
        if x.dim() == 3:
            x = x.unsqueeze(0)
            out = nn.functional.interpolate(x, size=self.size, mode='bilinear', align_corners=False)
            return out.squeeze(0)
        else:
            return nn.functional.interpolate(x, size=self.size, mode='bilinear', align_corners=False)

class Normalize(nn.Module):
    """
    Normalize an image tensor (assumes float values in [0,1]) using given mean and std.
    """
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        
    def forward(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

class ToFloatAndScale(nn.Module):
    """
    Convert the image tensor to float and scale pixel values from [0, 255] to [0, 1].
    """
    def forward(self, x):
        return x.float() / 255.0

class TransformCompose:
    """
    Compose several transformations sequentially.
    """
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

# Define a transformation composition matching ResNet18's requirements:
# Final image size: 3x224x224, float in [0,1], then normalized.
transform_fn = TransformCompose([
    Resize((224, 224)),
    ToFloatAndScale(),  
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_device(verbose=True):
    """
    Returns CUDA device if available, otherwise CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
    return device

def build_model(device):
    """
    Loads a pretrained ResNet18 and adapts it for binary classification.
    """
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # Binary classifier
    return model.to(device)

def train(model, train_loader, criterion, optimizer, device, num_epochs, transform_fn):
    """
    Main training loop.
    """
    print("Starting Training...")
    for epoch in range(num_epochs):
        # Switch to training mode
        train_loader.set("train")
        model.train()
        running_loss = 0.0
        total_samples = 0
        correct_preds = 0
        
        for batch in train_loader:
            # Now batch is a dictionary with keys 'image', 'label', etc.
            # Apply our transform_fn to each image in the batch
            images = torch.stack([transform_fn(img) for img in batch["image"]]).to(device)
            labels = torch.tensor([int(lbl) for lbl in batch["label"]], device=device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += (preds == labels).sum().item()
            total_samples += labels.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = correct_preds / total_samples * 100
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")
    
    print("Training complete.")

@torch.no_grad()
def evaluate(model, dataset, mode, batch_size, device, transform_fn):
    """
    Evaluate the model on `mode` subset (either 'test' or 'validation').
    """
    # Evaluate model on the specified dataset mode: "test" or "validation"
    eval_loader = FaceDataLoader(dataset, batch_size=batch_size, shuffle=False)
    eval_loader.set(mode)
    model.eval()
    total_samples = 0
    correct_preds = 0
    
    for batch in eval_loader:
        images = torch.stack([transform_fn(img) for img in batch["image"]]).to(device)
        labels = torch.tensor([int(lbl) for lbl in batch["label"]], device=device)
        
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct_preds += (preds == labels).sum().item()
        total_samples += labels.size(0)
    
    if total_samples > 0:
        acc = correct_preds / total_samples * 100
        print(f"{mode.capitalize()} Accuracy: {acc:.2f}%")
    else:
        print(f"No {mode} data available for evaluation.")

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train ResNet18 on face dataset.")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Root directory containing train/test/validation folders.")
    parser.add_argument("--epochs", type=int, default=2, 
                        help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, 
                        help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=0.001, 
                        help="Learning rate.")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Set up device
    device = get_device()
    
    # Load the dataset
    dataset = FaceDataset(args.data_dir, preload=False)
    
    # Create a DataLoader using the custom FaceDataLoader
    train_loader = FaceDataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Build the model
    model = build_model(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train the model
    train(model, train_loader, criterion, optimizer, device, 
          num_epochs=args.epochs, transform_fn=transform_fn)
    
    # Evaluate on test set
    evaluate(model, dataset, "test", args.batch_size, device, transform_fn)
    
    # Evaluate on validation set
    evaluate(model, dataset, "validation", args.batch_size, device, transform_fn)
