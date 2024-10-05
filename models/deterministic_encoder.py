class DeterministicEncoder(torch.nn.Module):
    def __init__(self, obs_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(obs_dim)
        self.fc = torch.nn.Linear(self.encoder.output_dim, latent_dim)
        self.output_dim = latent_dim

    def forward(self, x):
        return self.fc(self.encoder(x))