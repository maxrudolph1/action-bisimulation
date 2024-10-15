
class Encoder(torch.nn.Module):
    def __init__(self, obs_dim, normalized=False, **kwargs):
        super().__init__()
        c, h, w = obs_dim
        self.normalized = normalized
        
        self.grid = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(0, h) / (h - 1), torch.arange(0, w) / (w - 1)
                ),
                dim=0,
            )
            * 2
            - 1
        ).cuda()
        
        
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=c + 2 if 'use_grid' in kwargs and kwargs['use_grid'] else c,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding="same",
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding="same",
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding="same",
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(
                in_channels=96,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            torch.nn.ReLU(),
        )
        self.kwargs = kwargs
        if 'use_grid' in kwargs and kwargs['use_grid']:
            self.output_dim = self.conv(torch.zeros([1, c + 2, h, w])).flatten().shape[0]
        else:
            self.output_dim = self.conv(torch.zeros([1, c, h, w])).flatten().shape[0]
        

    def forward(self, obs):
        obs = torch.as_tensor(obs, device="cuda")

        if len(obs.shape) == 5:
            obs = obs.flatten(1, 2)
        if obs.dtype == torch.uint8:
            obs = obs / 127.5 - 1
        grid_expand = self.grid.expand(obs.shape[0], -1, -1, -1)

        if 'use_grid' in self.kwargs and self.kwargs['use_grid']:
            combined = torch.cat([obs, grid_expand], dim=1)
        else:
            combined = obs

        z = self.conv(combined).flatten(start_dim=1)
        return  z