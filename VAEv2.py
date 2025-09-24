import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import trange, tqdm
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as Datasets

import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils as vutils

import os
import random
import numpy as np
import math
from IPython.display import clear_output
import matplotlib.pyplot as plt
from PIL import Image

from datasets import load_dataset


# - - - - - - Dataset - - - - - -
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    ])


# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("Armaggheddon/lego_minifigure_captions", split='train')

class HFImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item['image']
        if self.transform:
            img = self.transform(img)
        return img


dataset = HFImageDataset(ds, transform=transform)

# - - - - - - Configuration - - - - - -

input_dim = 128*128*3
h_dim = 200
z_dim = 20


batch_size = 64
lr_rate = 1e-3
num_epochs = 5
latent_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

# - - - - - - Model - - - - - -

#input img -> hidden dim -> mean, std -> reparametrization trick -> decoder -> output img
class VAE(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # [3,128,128] -> [32,64,64]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # [64,64] -> [64,32,32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # -> [128,16,16]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),# -> [256,8,8]
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256*8*8, z_dim)
        self.fc_logvar = nn.Linear(256*8*8, z_dim)

        # Decoder
        self.fc = nn.Linear(z_dim, 256*8*8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # -> [128,16,16]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # -> [64,32,32]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # -> [32,64,64]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),    # -> [3,128,128]
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.fc(z).view(-1, 256, 8, 8)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar





model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr_rate)



if __name__ == "__main__":

    # - - - - - - Loss function - - - - - -

    def loss_fn(x_recon, x, mu, logvar):
        recon_loss = F.mse_loss(x_recon, x, reduction="sum")
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div



    # - - - - - - Training Loop - - - - - -

    for epoch in range(num_epochs):
        loop = tqdm(train_loader)
        for i, x in enumerate(loop):

            #forward pass
            x = x.to(device)
            x_recon, mu, logvar = model(x)

            #loss
            loss = loss_fn(x_recon, x, mu, logvar)

            #backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix({'Train loss': loss.item()})

    dataiter = iter(test_loader)
    test_images = next(dataiter)
    print(test_images.shape)

    model.eval()
    recon_data, mu, sigma = model(test_images.to(device))

    plt.figure(figsize = (20, 10))
    out = vutils.make_grid(test_images[20:28], normalize=True)
    plt.imshow(out.numpy().transpose((1, 2, 0)))
    plt.show()

    plt.figure(figsize = (20, 10))
    out = vutils.make_grid(recon_data.detach().cpu()[20:28], normalize=True)
    plt.imshow(out.numpy().transpose((1, 2, 0)))
    plt.show()

    rand_samp = model.decode(mu+0.1*torch.randn_like(mu))
    plt.figure(figsize = (20, 10))
    out = vutils.make_grid(rand_samp.detach().cpu()[20:28], normalize=True)
    plt.imshow(out.numpy().transpose((1, 2, 0)))
    plt.show()

    rand_samp = model.decode(torch.randn_like(mu))
    plt.figure(figsize = (20, 10))
    out = vutils.make_grid(rand_samp.detach().cpu()[20:28], normalize=True)
    plt.imshow(out.numpy().transpose((1, 2, 0)))
    plt.show()



    # - - - - - - Saving - - - - - -

    torch.save(model.state_dict(), "vae.pth")
    print("Saved to vae.pth")



    # - - - - - - Style transfer - - - - - -
    def lego_image(model, img, device):
        model.eval()
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            recon, _, _ = model(x)
        return recon.cpu().view(3, 128, 128)
