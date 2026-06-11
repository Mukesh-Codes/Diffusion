import argparse
import torch
import matplotlib.pyplot as plt
import os

from data.ou_data import get_dataloader
from schedules import StandardSchedule, Ellipsoid
from samplers import DDPMSampler
from models.unet1d import UNet1D
from train import train_model
import soundfile as sf
import os

def plot_spectrogram(data, sample_rate=16000, filename="spectrogram.png"):
    print("Generating spectrograms...")

    # Create a tall stack of plots for however many clips we generated
    fig, axes = plt.subplots(len(data), 1, figsize=(10, 2.5 * len(data)))

    # Handle the math if we only generate 1 clip
    if len(data) == 1: axes = [axes]

    for i, waveform in enumerate(data):
        # matplotlib's specgram does all the heavy Fourier math for us!
        axes[i].specgram(waveform, Fs=sample_rate, cmap='magma')
        axes[i].set_ylabel('Frequency (Hz)')
        axes[i].set_title(f'Trajectory {i+1} Spectrogram')

    plt.xlabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print(f"📊 Saved {filename} to disk!")

def save_audio(data, sample_rate=16000, prefix="generated_sound"):
    """Converts 1D numpy arrays into playable .wav files!"""
    print("Saving audio files...")
    for i in range(len(data)):
        waveform = data[i]
        filename = f"{prefix}_{i+1}.wav"
        sf.write(filename, waveform, sample_rate)
        print(f"🎵 Saved {filename} to disk!")


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
    parser.add_argument("--dataset", choices=["ou", "audioset"], default="ou", help="Which dataset to use")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--schedule", choices=["standard", "elliptical"], default="standard", required=False, help = "standard or elliptical")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--samples", type=int, default=5, help="Number of trajectories to generate")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = 4096 if args.dataset == "audioset" else 100
    if args.schedule == "elliptical":
        schedule = Ellipsoid(num_steps=1000, d=seq_len, beta_start=1e-4, beta_end=0.02, device=device)
    else:
        schedule = StandardSchedule(num_steps=1000, beta_start=1e-4, beta_end=0.02, device=device)
    schedule = StandardSchedule(num_steps=1000, beta_start=1e-4, beta_end=0.02, device=device)
    model = UNet1D().to(device)
    weights_path = f"unet_{args.schedule}_weights.pt"
    if args.mode == "train":
        dataloader = get_dataloader(dataset_name=args.dataset, batch_size=args.batch_size)
        train_model(model, schedule, dataloader, args.epochs, device, save_path=weights_path)
    elif args.mode == "sample":
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Cannot find {weights_path}. You must run --mode train first!")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Model weights loaded.")
        sampler = DDPMSampler(schedule, model, device=device)
        print(f"Generating {args.samples} trajectories... This may take a moment.")
        generated_data = sampler.sample(num_samples=args.samples, seq_len=seq_len)
        generated_data = generated_data.cpu().squeeze(1).numpy()
        plot_trajectories(generated_data, filename=f"generated_{args.schedule}.png")
        if args.dataset == "audioset":
            save_audio(generated_data, sample_rate=16000, prefix=f"audio_{args.schedule}")
            plot_spectrogram(generated_data, sample_rate=16000, filename=f"spectrogram_{args.schedule}.png")

if __name__ == "__main__":
    main()