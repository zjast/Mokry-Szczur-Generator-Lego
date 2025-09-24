import os
import torch
from VAEv2 import VAE
from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# - - - - - - Model loading - - - - - -
model = VAE().to(device)
model.load_state_dict(torch.load("vae.pth", map_location=device))
model.eval()


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

images = ["StandingMan", "StandingWoman", "StandingBoy", "StandingGirl", "Kot", "Chłop", "Kwiat"]

recon_data = []
org_data = []

with torch.no_grad():
    for i in images:
        img = Image.open(f"Tests/Original/{i}.jpg").convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        recon, _, _ = model(x)

        recon_data.append(recon.squeeze(0).cpu())
        org_data.append(x.squeeze(0))


all_imgs = org_data + recon_data
grid = vutils.make_grid(all_imgs, nrow=len(images), padding=10, normalize=True)

plt.figure(figsize=(20, 10))
plt.imshow(grid.permute(1, 2, 0).numpy())
plt.axis('off')
plt.show()