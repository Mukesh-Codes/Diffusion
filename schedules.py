import torch
from torch.distributions.multivariate_normal import MultivariateNormal

class StandardSchedule():
    def __init__(self, num_steps, beta_start, beta_end, device="cpu"):
        self.device=device
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def sample(self, data):
        d = data.shape[0]
        l = data.shape[2]
        mean = torch.zeros(l, device=self.device)
        cov_matrix = torch.eye(l, device=self.device)
        m = MultivariateNormal(loc=mean, covariance_matrix=cov_matrix)
        epsilon = m.sample((d,)).unsqueeze(1)
        k = torch.randint(0, self.num_steps, size = (d,), device=self.device)
        sqrt_alpha_k = self.sqrt_alphas_cumprod.to(self.device)[k]
        sqrt_one_minus_k = self.sqrt_one_minus_alphas_cumprod.to(self.device)[k]
        sqrt_alpha_k = sqrt_alpha_k.view(-1, 1, 1)
        sqrt_one_minus_k = sqrt_one_minus_k.view(-1, 1, 1)
        x_k = (sqrt_alpha_k * data) + (sqrt_one_minus_k * epsilon)
        return x_k, epsilon, k


class Ellipsoid():
    def __init__(self, num_steps, d, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.device = device
        self.num_steps = num_steps
        self.d = d
        base_betas = torch.linspace(beta_start, beta_end, num_steps, device=device).unsqueeze(1)
        x = torch.linspace(-1.0, 1.0, d, device=device)
        spatial_modifier = (1.0 + torch.sqrt(1.0 - x**2)).unsqueeze(0)
        self.betas = base_betas * spatial_modifier
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def sample(self, data):
        b = data.shape[0]
        epsilon = torch.randn_like(data)
        k = torch.randint(0, self.num_steps, size=(b,), device=self.device)
        sqrt_alpha_k = self.sqrt_alphas_cumprod[k]
        sqrt_one_minus_k = self.sqrt_one_minus_alphas_cumprod[k]
        sqrt_alpha_k = sqrt_alpha_k.unsqueeze(1)
        sqrt_one_minus_k = sqrt_one_minus_k.unsqueeze(1)
        x_k = (sqrt_alpha_k * data) + (sqrt_one_minus_k * epsilon)
        return x_k, epsilon, k
