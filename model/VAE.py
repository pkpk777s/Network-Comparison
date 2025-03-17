import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 256
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 100, 100, 3

class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(Encoder, self).__init__()
        self.activation = nn.SiLU()
        # (3, 100, 100) -> (16, 100, 100)
        self.conv1 = nn.Conv2d(IMG_CHANNELS, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # (16, 100, 100) -> (32, 50, 50)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # (32, 50, 50) -> (64, 25, 25)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.flatten = nn.Flatten()
        # Flattened feature size 40,000.
        self.fc_mean = nn.Linear(64 * 25 * 25, latent_dim)
        self.fc_logvar = nn.Linear(64 * 25 * 25, latent_dim)
        
    def process(self, x):
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
    def __init__(self, latent_dim=LATENT_DIM):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS)
        self.activation = nn.SiLU()
        self.out_activation = nn.Sigmoid() 
        
    def process(self, z):
        x = self.fc1(z)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.out_activation(x)
        x = x.view(-1, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def process(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    # Flatten the images
    x = x.view(x.size(0), -1)
    recon_x = recon_x.view(recon_x.size(0), -1)
    # Reconstruction loss: binary cross entropy
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Example usage:
if __name__ == "__main__":
    batch_size = 4
    dummy_input = torch.rand(batch_size, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    
    model = VAE(LATENT_DIM)
    reconstruction, mu, logvar = model(dummy_input)
    
    loss = vae_loss(reconstruction, dummy_input, mu, logvar)
    print("Loss:", loss.item())
