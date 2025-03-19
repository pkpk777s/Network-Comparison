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
    def __init__(self, latent_dim=LATENT_DIM):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * 25 * 25)
        
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        
        self.deconv3 = nn.Conv2d(16, IMG_CHANNELS, kernel_size=3, padding=1)

        self.activation = nn.SiLU()
        self.out_activation = nn.Sigmoid() 
        
    def forward(self, z):
        x = self.fc(z)
        x = self.activation(x)
        
        x = x.view(-1, 64, 25, 25)
        
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
    def __init__(self, latent_dim=LATENT_DIM):
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
    cls_loss = F.binary_cross_entropy(fake_prob, gt_f_t, reduction='mean')

    Total_loss = BCE + KLD + cls_loss
    return Total_loss

# Example usage:
if __name__ == "__main__":
    base_dir = "./data"
    batch_size = 64
    dataset = FaceDataset(base_dir, preload=False)
    dataLoader = FaceDataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = VAE(LATENT_DIM)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataLoader.set("train")

    num_epochs = 25 

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in dataLoader:
            optimizer.zero_grad()
            
            img = batch['image']
            gt_f_t = batch['label']
            reconstruction, mu, logvar, fake_prob = model(img)
            
            loss = vae_loss(reconstruction, img, mu, logvar, fake_prob, gt_f_t)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    # save model later uncomment this when training
    # torch.save(model.state_dict(), "VAE_model.pth")