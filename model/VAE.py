import torch
import torch.nn as nn
import torch.nn.functional as F
from Dataloader import FaceDataLoader, FaceDataset
from torch.nn.init import xavier_uniform_
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self, latent_dim=64, IMG_CHANNELS=3):
        super(Encoder, self).__init__()
        self.activation = nn.SiLU()
        # conv1: (3,299,299) -> (16,299,299)
        self.conv1 = nn.Conv2d(IMG_CHANNELS, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # conv2: (16,299,299) -> (32,150,150)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # conv3: (32,150,150) -> (64,75,75)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.flatten = nn.Flatten()
        
        self.fc_mean = nn.Linear(64 * 75 * 75, latent_dim)
        self.fc_logvar = nn.Linear(64 * 75 * 75, latent_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        
        x = self.flatten(x)
        z_mean = self.fc_mean(x)
        z_logvar = self.fc_logvar(x)
        return z_mean, z_logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=64, IMG_CHANNELS=3):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * 75 * 75)
        
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        
        self.deconv3 = nn.Conv2d(16, IMG_CHANNELS, kernel_size=3, padding=1)

        self.activation = nn.SiLU()
        self.out_activation = nn.Sigmoid() 
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, z):
        x = self.fc(z)
        x = self.activation(x)
        
        x = x.view(-1, 64, 75, 75)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        x = self.deconv3(x)
        x = self.out_activation(x)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  
        )
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        fake_prob = self.classifier(z)
        return reconstruction, mu, logvar, fake_prob

def vae_loss(recon_x, x, mu, logvar, fake_prob, gt_f_t):
    # Flatten the images
    x = x.view(x.size(0), -1)
    recon_x = recon_x.view(recon_x.size(0), -1)
    # Reconstruction loss: binary cross entropy
    BCE = F.binary_cross_entropy(recon_x, x, reduction='mean')
    # KL divergence
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # Classification loss
    cls_loss = F.binary_cross_entropy(fake_prob, gt_f_t.unsqueeze(1).float(), reduction='mean')
    
    Total_loss = BCE + KLD + cls_loss
    # Total_loss = cls_loss
    return Total_loss

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

# Example usage:
if __name__ == "__main__":
    LATENT_DIM = 64
    base_dir = "./data"
    batch_size = 16
    dataset = FaceDataset(base_dir, preload=False)
    dataLoader = FaceDataLoader(dataset, batch_size=batch_size, shuffle=True)
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
    loss_plot_path = os.path.join(save_dir, 'vae_loss_plot.png')
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")
    
    # Save model
    model_path = os.path.join(save_dir, 'VAE_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    