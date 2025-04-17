import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import numpy as np

from VAE import VAE, vae_loss 
from Preprocessor import Preprocessor
from Dataloader import FaceDataset

class PreprocessedFaceDataset(Dataset):

    def __init__(self, base_dir, preprocessor: Preprocessor, preload=False):
        self.raw_dataset = FaceDataset(base_dir, preload=preload)
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        sample = self.raw_dataset[idx]
        img = sample['image']
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif isinstance(img, torch.Tensor):
            img = to_pil_image(img)

        proc = self.preprocessor(img)
        img_pil = proc['multi_scale_images']['scale_1.0']
        img_tensor = to_tensor(img_pil)
        label = sample['label']
        return {'image': img_tensor, 'label': label}

if __name__ == '__main__':
    base_dir   = './data'
    batch_size = 64
    latent_dim = 64
    num_epochs = 25
    lr         = 1e-4

    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocessor = Preprocessor(scales=[0.5, 1.0, 1.5], blur_radius=2)
    train_ds     = PreprocessedFaceDataset(base_dir, preprocessor, preload=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model     = VAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader, start=1):
            img   = batch['image'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()
            recon, mu, logvar, fake_prob = model(img)
            loss = vae_loss(recon, img, mu, logvar, fake_prob, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Epoch {epoch}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} "
                  f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"\n--- Epoch {epoch} Summary ---\nAverage Loss: {avg_loss:.4f}\n")

    torch.save(model.state_dict(), 'preprocessed_vae.pth')
    print("Model saved to preprocessed_vae.pth")
