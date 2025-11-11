"""
scripts/train_diffusion.py
Minimal DDPM-style diffusion training on MedMNIST (PathMNIST / OrganMNIST).
Designed for small GPUs (e.g., RTX 3050). Produces saved sample grids and checkpoints.
"""

import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image

# medmnist
from medmnist import INFO, PathMNIST, OrganMNIST

# -------------------------
# Utilities / UNet (small)
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.AvgPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.net(x)

class TinyUNet(nn.Module):
    def __init__(self, channels=1, base_ch=32):
        super().__init__()
        self.inc = DoubleConv(channels, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.mid = DoubleConv(base_ch*4, base_ch*4)
        self.up2 = Up(base_ch*4, base_ch*2)
        self.up1 = Up(base_ch*2, base_ch)
        self.outc = nn.Conv2d(base_ch, channels, 1)
    def forward(self, x, t_emb=None):
        # t_emb is unused in this tiny model; later inject with FiLM or cross-attention.
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        xm = self.mid(x3)
        xu = self.up2(xm)
        xu = xu + x2  # skip
        xu = self.up1(xu)
        xu = xu + x1  # skip
        out = self.outc(xu)
        return out

# -------------------------
# Diffusion utilities
# -------------------------
def linear_beta_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 2e-2
    return torch.linspace(beta_start, beta_end, timesteps)

class Diffusion:
    def __init__(self, timesteps=1000, device='cpu'):
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise=None):
        # x_t = sqrt(alpha_hat_t) * x0 + sqrt(1 - alpha_hat_t) * noise
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_hat = self.alpha_hat[t].sqrt().view(-1,1,1,1)
        sqrt_one_minus = (1 - self.alpha_hat[t]).sqrt().view(-1,1,1,1)
        return sqrt_alpha_hat * x_start + sqrt_one_minus * noise

    def predict_start_from_noise(self, x_t, t, noise):
        return (x_t - (1 - self.alpha_hat[t]).sqrt().view(-1,1,1,1) * noise) / self.alpha_hat[t].sqrt().view(-1,1,1,1)

# -------------------------
# Training loop
# -------------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'samples').mkdir(exist_ok=True)
    (out_dir / 'checkpoints').mkdir(exist_ok=True)

    # Dataset selection
    info = INFO[args.dataset]
    DataClass = PathMNIST if args.dataset == 'PathMNIST' else OrganMNIST
    transform = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor()])
    train_ds = DataClass(split='train', transform=transform, download=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Model + diffusion
    model = TinyUNet(channels=1 if info['n_channels'] == 1 else 3, base_ch=args.base_ch).to(device)
    diffusion = Diffusion(timesteps=args.timesteps, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            imgs, labels = batch[0].to(device), batch[1].to(device)  # medmnist returns (img, label)
            imgs = imgs.float()  # in [0,1]
            imgs = imgs * 2. - 1.  # scale to [-1,1] for stability

            bsz = imgs.shape[0]
            t = torch.randint(0, args.timesteps, (bsz,), device=device).long()
            noise = torch.randn_like(imgs)
            x_t = diffusion.q_sample(imgs, t, noise=noise)

            pred_noise = model(x_t)  # predicted noise
            loss = nn.MSELoss()(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            steps += 1
            pbar.set_postfix(loss=float(loss.detach()))

            # save samples
            if steps % args.save_every == 0:
                model.eval()
                with torch.no_grad():
                    # sample from pure noise by simple single-step pseudo-reverse (for demo)
                    n = min(16, imgs.size(0))
                    z = torch.randn(n, imgs.size(1), args.img_size, args.img_size, device=device)
                    # simple (not proper ancestral sampling) iterative denoising
                    x = z
                    for time_step in reversed(range(0, args.timesteps, max(1, args.timesteps//50))):
                        t_batch = torch.full((n,), time_step, device=device, dtype=torch.long)
                        pred_n = model(x)
                        # model predicts noise; perform a simple update
                        coef = diffusion.betas[time_step].sqrt()
                        x = (x - coef * pred_n)
                        # clamp for stability
                        x = x.clamp(-1,1)
                    samples = (x + 1) / 2.0  # back to [0,1]
                    grid = make_grid(samples.cpu(), nrow=4, normalize=False)
                    save_image(grid, out_dir / 'samples' / f'step_{steps}.png')
                # save checkpoint
                torch.save({'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'steps': steps}, out_dir / 'checkpoints' / f'ckpt_{steps}.pt')
                model.train()

        # end epoch

    print("Training finished. Samples and checkpoints saved to", out_dir)

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PathMNIST', choices=['PathMNIST', 'OrganMNIST'])
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--timesteps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--base_ch', type=int, default=32)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--output_dir', type=str, default='results/diffusion_run')
    args = parser.parse_args()
    train(args)
