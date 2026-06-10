import argparse
import torch
import matplotlib.pyplot as plt
import os

from data.ou_data import get_dataloader
from schedules import StandardSchedule
from samplers import DDPMSampler
from models.unet1d import UNet1D
from train import train_model

def plot_trajectories(data, filename="generated.png"):
    plt.figure(figsize=(10, 6))
    for i in range(len(data)):
        plt.plot(data[i], label=f"Trajectory {i+1}")
    plt.title("Diffusion Generated Time-Series")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="1D Diffusion Command Center")
    parser.add_argument("--mode", choices=["train", "sample"], required=True, help="Train the model or generate samples")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--samples", type=int, default=5, help="Number of trajectories to generate")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    schedule = StandardSchedule(num_steps=1000, beta_start=1e-4, beta_end=0.02, device=device)
    model = UNet1D().to(device)
    weights_path = "unet_weights.pt"
    if args.mode == "train":
        dataloader = get_dataloader(batch_size=args.batch_size, count=20000)
        train_model(model, schedule, dataloader, args.epochs, device, save_path=weights_path)
    elif args.mode == "sample":
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Cannot find {weights_path}. You must run --mode train first!")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Model weights loaded.")
        sampler = DDPMSampler(schedule, model, device=device)
        print(f"Generating {args.samples} trajectories... This may take a moment.")
        generated_data = sampler.sample(num_samples=args.samples, seq_len=100)
        generated_data = generated_data.cpu().squeeze(1).numpy()
        plot_trajectories(generated_data)

if __name__ == "__main__":
    main()