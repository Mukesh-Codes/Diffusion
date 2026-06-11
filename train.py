import torch
import torch.nn.functional as F
from torch.optim import Adam
import os

def train_model(model, schedule, dataloader, epochs, device, save_path="weights.pt"):
    print(f"Starting training for {epochs} epochs on {device}")
    optimizer = Adam(model.parameters(), lr=1e-4)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches_processed=0
        for step, batch in enumerate(dataloader):
            if step>100:
                break
            x_0 = batch.to(device)
            optimizer.zero_grad()
            x_k, true_noise, k = schedule.sample(x_0)
            predicted_noise = model(x_k, k)
            loss = F.mse_loss(predicted_noise, true_noise)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches_processed+=1
        if batches_processed > 0:
            avg_loss = epoch_loss / batches_processed
        else:
            avg_loss = 0.0
        print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {avg_loss:.5f}")

    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Weights saved to {save_path}")