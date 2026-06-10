import torch

class DDPMSampler:
    def __init__(self, schedule, model, device="cpu"):
        self.schedule = schedule
        self.model = model
        self.device = device

    @torch.no_grad()
    def sample(self, num_samples, seq_len):
        self.model.eval()
        x = torch.randn((num_samples, 1, seq_len), device=self.device)
        for k in reversed(range(self.schedule.num_steps)):
            k_tensor = torch.full((num_samples,), k, device=self.device, dtype=torch.long)
            predicted_noise = self.model(x, k_tensor)
            alpha_k = self.schedule.alphas[k]
            sqrt_one_minus_cumprod_k = self.schedule.sqrt_one_minus_alphas_cumprod[k]
            beta_k = self.schedule.betas[k]
            x_mean = (1.0 / torch.sqrt(alpha_k)) * (
                x - ((1.0 - alpha_k) / sqrt_one_minus_cumprod_k) * predicted_noise
            )
            if k > 0:
                fresh_noise = torch.randn_like(x)
                x = x_mean + torch.sqrt(beta_k) * fresh_noise
            else:
                x = x_mean
        return x