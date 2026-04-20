import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import math

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

img_size = 28
channels = 1
timesteps = 100

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, channels)
        self.act = nn.SiLU()

    def forward(self, x, t):
        h = self.act(self.conv1(x))
        h = h + self.time_mlp(t)[:, :, None, None]
        h = self.act(self.conv2(h))
        return x + h

class Unet(nn.Module):
    def __init__(self, channels=64, time_dim=128):
        super().__init__()
        self.time_embed = SinusoidalPositionEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )

        self.conv_in = nn.Conv2d(1, channels, 3, padding=1)

        self.down1 = ResidualBlock(channels, time_dim)
        self.down2 = ResidualBlock(channels * 2, time_dim)
        self.down3 = ResidualBlock(channels * 2, time_dim)

        self.conv1 = nn.Conv2d(channels, channels * 2, 3, 2, 1)
        self.conv2 = nn.Conv2d(channels * 2, channels * 2, 3, 2, 1)

        self.up1 = nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(channels, channels, 4, 2, 1)

        self.up1_res = ResidualBlock(channels, time_dim)
        self.up2_res = ResidualBlock(channels, time_dim)

        self.conv_out = nn.Conv2d(channels, channels, 3, padding=1)
        self.final = nn.Conv2d(channels, 1, 1)
        self.act = nn.SiLU()

    def forward(self, x, t):
        t = self.time_embed(t)
        t = self.time_mlp(t)

        x = self.conv_in(x)

        x = self.down1(x, t)
        x = self.conv1(x)
        x = self.act(x)

        x = self.down2(x, t)
        x = self.conv2(x)
        x = self.act(x)

        x = self.up1(x)
        x = self.up1_res(x, t)
        x = self.up2(x)
        x = self.up2_res(x, t)

        x = self.conv_out(x)
        x = self.act(x)
        x = self.final(x)

        return x

class Diffusion:
    def __init__(self, timesteps=100, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1 - betas
        alphas_cum = torch.cumprod(alphas, dim=0)
        self.alphas_cum = alphas_cum

    def noise_images(self, images, t):
        batch = images.size(0)
        alphas_cum = self.alphas_cum.to(images.device)
        sqrt_alpha_cum = torch.sqrt(alphas_cum[t])[:, None, None, None]
        sqrt_one_minus_alpha_cum = torch.sqrt(1 - alphas_cum[t])[:, None, None, None]
        noise = torch.randn_like(images)
        noisy = sqrt_alpha_cum * images + sqrt_one_minus_alpha_cum * noise
        return noisy, noise

def train(epochs=20, batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = Unet(channels=64, time_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    diffusion = Diffusion(timesteps=timesteps)

    os.makedirs("weights", exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device) * 2 - 1

            batch = imgs.size(0)
            t = torch.randint(0, timesteps, (batch,), device=device)

            noisy_imgs, noise = diffusion.noise_images(imgs, t)

            optimizer.zero_grad()
            predicted = model(noisy_imgs, t)
            loss = nn.functional.mse_loss(predicted, noise)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"weights/diffusion_{epoch+1}.pth")
            print(f"Saved weights/diffusion_{epoch+1}.pth")

    torch.save(model.state_dict(), "weights/diffusion_final.pth")
    print("Training complete. Saved to weights/diffusion_final.pth")

if __name__ == "__main__":
    train()