import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import numpy as np

from VAE import VAE, vae_loss 
from Preprocessor import Preprocessor
from Dataloader import FaceDataLoader, FaceDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

class PreprocessedFaceDataset(FaceDataset):
    def __init__(self, base_dir, preprocessor: Preprocessor, preload=False):
        # Call the parent class constructor with proper arguments
        super().__init__(base_dir, preload)
        self.preprocessor = preprocessor

    def __getitem__(self, idx):
        entry = super().__getitem__(idx)
        img = entry['image']
        
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif isinstance(img, torch.Tensor):
            img = to_pil_image(img)

        # Apply preprocessing
        proc = self.preprocessor(img)
        img_pil = proc['multi_scale_images']['scale_1.0']
        img_tensor = to_tensor(img_pil)
        
        # Return the processed image and label
        return {'image': img_tensor, 'label': entry['label']}
    
def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            img = batch['image'].float() / 255.0
            gt_f_t = batch['label']
            reconstruction, mu, logvar, fake_prob = model(img)
            loss = vae_loss(reconstruction, img, mu, logvar, fake_prob, gt_f_t)
            total_loss += loss.item()
            total_batches += 1
    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    model.train()
    return avg_loss

# if __name__ == '__main__':
#     base_dir   = './data'
#     batch_size = 64
#     latent_dim = 64
#     num_epochs = 25

#     device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     preprocessor = Preprocessor(scales=[0.5, 1.0, 1.5], blur_radius=2)
#     train_ds     = PreprocessedFaceDataset(base_dir, preprocessor, preload=False)
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

#     model     = VAE(latent_dim).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#     model.train()
#     for epoch in range(1, num_epochs + 1):
#         total_loss = 0.0
#         for batch_idx, batch in enumerate(train_loader, start=1):
#             img   = batch['image'].to(device)
#             label = batch['label'].to(device)

#             optimizer.zero_grad()
#             recon, mu, logvar, fake_prob = model(img)
#             loss = vae_loss(recon, img, mu, logvar, fake_prob, label)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             print(f"Epoch {epoch}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} "
#                   f"Loss: {loss.item():.4f}")

#         avg_loss = total_loss / len(train_loader)
#         print(f"\n--- Epoch {epoch} Summary ---\nAverage Loss: {avg_loss:.4f}\n")

#     torch.save(model.state_dict(), 'preprocessed_vae.pth')
#     print("Model saved to preprocessed_vae.pth")

# Example usage:
if __name__ == "__main__":
    LATENT_DIM = 64
    base_dir = "./data"
    batch_size = 16
    preprocessor = Preprocessor(scales=[0.5, 1.0, 1.5], blur_radius=2)
    dataset = PreprocessedFaceDataset(base_dir, preprocessor, preload=False)
    dataLoader = FaceDataLoader(dataset, batch_size=4, shuffle=True)
    model = VAE(LATENT_DIM)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,  # Reduce LR by half when triggered
        patience=2,  # Wait for 2 epochs without improvement before reducing
        verbose=True
    )
    dataLoader.set("train")
    
    num_epochs = 64 

    save_dir = "../VAE_result"
    os.makedirs(save_dir, exist_ok=True)
    
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Create progress bar for this epoch
        pbar = tqdm(dataLoader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            img = batch['image'].float() / 255.0
            gt_f_t = batch['label']
            
            reconstruction, mu, logvar, fake_prob = model(img)
            loss = vae_loss(reconstruction, img, mu, logvar, fake_prob, gt_f_t)
            
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            total_loss += current_loss
            num_batches += 1
            
            # Update progress bar with current batch loss
            pbar.set_postfix({"Loss": f"{current_loss:.4f}"})
        
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_loss)
        
        # Evaluate on validation set at the end of each epoch
        dataLoader.set("validation")
        val_loss = evaluate_model(model, dataLoader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Set back to training mode and data
        dataLoader.set("train")
    
    # Evaluate on Test set after training
    dataLoader.set("test")
    test_loss = evaluate_model(model, dataLoader)
    print(f"Final Test Loss: {test_loss:.4f}")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Add markers for min values
    min_train_epoch = np.argmin(train_losses) + 1
    min_val_epoch = np.argmin(val_losses) + 1
    min_train_loss = min(train_losses)
    min_val_loss = min(val_losses)
    
    plt.annotate(f'Min: {min_train_loss:.4f}', 
                xy=(min_train_epoch, min_train_loss), 
                xytext=(min_train_epoch+1, min_train_loss+0.5),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.annotate(f'Min: {min_val_loss:.4f}', 
                xy=(min_val_epoch, min_val_loss), 
                xytext=(min_val_epoch+1, min_val_loss+0.5),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Save the figure
    loss_plot_path = os.path.join(save_dir, 'Preprocessed_vae_loss_plot.png')
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")
    
    # Save model
    model_path = os.path.join(save_dir, 'Preprocessed_VAE_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    