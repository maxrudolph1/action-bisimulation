class ForwardModel(torch.nn.Module):
    fw_model_step_loss = None

    def __init__(self, obs_shape, action_dim, args=None):
        super().__init__()
        self.forward_model = nets.ForwardNet(obs_shape, action_dim).cuda()
        self.optimizer = torch.optim.Adam(
            list(self.forward_model.parameters()),
            lr=1e-4,
        )
        
    def train_step(self, batch):
        obs = torch.as_tensor(batch["obs"], device="cuda")
        act = torch.as_tensor(batch["action"], device="cuda")
        obs_next = torch.as_tensor(batch["obs_next"], device="cuda")
        self.forward_model(obs, act)
        forward_loss = F.mse_loss(
            self.forward_model(obs, act),
            obs_next,
        )
        step_loss = forward_loss
        self.optimizer.zero_grad()
        forward_loss.backward()
        self.optimizer.step()

        return {"forward_loss": forward_loss.detach().item()}, self.forward_model(obs, act).detach()
