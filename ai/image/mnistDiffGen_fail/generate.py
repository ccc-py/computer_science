import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import os
import math
import argparse

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

def generate(weights_path="weights/diffusion_final.pth", num_images=16, output_dir="output"):
    model = Unet(channels=64, time_dim=128).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    betas = torch.linspace(0.0001, 0.02, timesteps)
    alphas = 1 - betas
    alphas_cum = torch.cumprod(alphas, dim=0)
    alphas_cum = alphas_cum.to(device)

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        x = torch.randn(num_images, 1, img_size, img_size, device=device)

        for i in range(timesteps - 1, 0, -1):
            t = torch.full((num_images,), i, device=device, dtype=torch.long)
            predicted = model(x, t)

            alpha = alphas[i]
            alpha_cum = alphas_cum[i]
            alpha_cum_prev = alphas_cum[i - 1] if i > 0 else torch.tensor(1.0, device=device)

            pred_original = (x - predicted * torch.sqrt(1 - alpha_cum)) / torch.sqrt(alpha_cum)

            coef1 = (1 - alpha_cum_prev).sqrt() * alpha.sqrt() / (1 - alpha_cum)
            coef2 = (1 - alpha).sqrt() * alpha_cum_prev.sqrt() / (1 - alpha_cum)

            x = coef1 * pred_original + coef2 * predicted

            if i > 0:
                noise = torch.randn_like(x)
                x = x + noise * torch.sqrt(1 - alpha) * 0

        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)

    for i, img in enumerate(x):
        save_image(img, f"{output_dir}/digit_{i:03d}.png")

    save_image(x, f"{output_dir}/grid.png", nrow=4, normalize=True)
    print(f"Generated {num_images} images in {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/diffusion_final.pth")
    parser.add_argument("--num", type=int, default=16)
    parser.add_argument("--output", default="output")
    args = parser.parse_args()

    generate(args.weights, args.num, args.output)