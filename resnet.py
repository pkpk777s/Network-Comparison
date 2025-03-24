import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

############################################
# Custom dataset and dataloader definitions
############################################

class FaceDataset(torch.utils.data.Dataset):
    """
    A custom dataset that reads images from directories:
      data/train, data/test, data/validation.
    Image filenames should begin with:
      'R_' for real (label 1) and 'F_' for fake (label 0).
    If preload is False, images are loaded on-demand.
    """
    def __init__(self, base_dir, preload=False):
        self.base_dir = base_dir
        self.preload = preload
        self.data = []
        
        modes = ["train", "test", "validation"]
        for mode in modes:
            mode_dir = os.path.join(base_dir, mode)
            if not os.path.isdir(mode_dir):
                continue
            for fname in os.listdir(mode_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(mode_dir, fname)
                    # Label is 1 (real) if filename does not start with 'F_', else 0 (fake)
                    label = 0 if fname.startswith("F_") else 1
                    if preload:
                        image = self.read_image(full_path)
                        self.data.append({"image": image, "label": label, "mode": mode})
                    else:
                        self.data.append({"path": full_path, "label": label, "mode": mode})
        # By default, set to train mode
        self.set_mode("train")
    
    def set_mode(self, mode):
        """Filter the dataset samples to a specific mode."""
        self.samples = [sample for sample in self.data if sample["mode"] == mode]
    
    def __len__(self):
        return len(self.samples)
    
    def read_image(self, path):
        """Load an image from disk and convert to a tensor (C,H,W) with uint8 values."""
        with Image.open(path) as img:
            # Ensure image is RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_array = np.array(img)
            # Change shape from (H,W,C) to (C,H,W)
            tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            return tensor
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if "image" not in sample:
            sample["image"] = self.read_image(sample["path"])
        return sample

class FaceDataLoader:
    """
    A simple wrapper around torch.utils.data.DataLoader that allows switching modes.
    """
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._create_loader()
    
    def _create_loader(self):
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )
    
    def set(self, mode):
        self.dataset.set_mode(mode)
        self._create_loader()
    
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self):
        return len(self.loader)

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

# Load the dataset
data_dir = "./data"  # Update this path as needed
dataset = FaceDataset(data_dir, preload=False)

# Create a DataLoader using the custom FaceDataLoader
batch_size = 4
train_loader = FaceDataLoader(dataset, batch_size=batch_size, shuffle=True)

# Prepare the ResNet18 model - adjust the final fully connected layer for binary classification
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # two output classes for binary classification
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define training parameters
num_epochs = 2  # Increase number of epochs as needed

def train():
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
            images = torch.stack([transform_fn(img) for img in batch["image"]])
            labels = torch.tensor([int(lbl) for lbl in batch["label"]])
            
            images = images.to(device)
            labels = labels.to(device)
            
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

def evaluate(mode="test"):
    # Evaluate model on the specified dataset mode: "test" or "validation"
    eval_loader = FaceDataLoader(dataset, batch_size=batch_size, shuffle=False)
    eval_loader.set(mode)
    model.eval()
    total_samples = 0
    correct_preds = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            images = torch.stack([transform_fn(img) for img in batch["image"]])
            labels = torch.tensor([int(lbl) for lbl in batch["label"]])
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct_preds += (preds == labels).sum().item()
            total_samples += labels.size(0)
    
    if total_samples > 0:
        acc = correct_preds / total_samples * 100
        print(f"{mode.capitalize()} Accuracy: {acc:.2f}%")
    else:
        print(f"No {mode} data available for evaluation.")

if __name__ == "__main__":
    train()
    # Evaluate on test set
    evaluate("test")
    # Evaluate on validation set
    evaluate("validation")
