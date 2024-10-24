from copy import deepcopy

import torch
import torch.nn.functional as F
from models import gen_model_nets


class EncoderReconstruction(torch.nn.Module):
    def __init__(
        self,
        obs_shape,
        act_shape=None,
        encoder_cfg=None,
        forward_cfg=None,
        inverse_cfg=None,
        # learning_rate=0.01,
        learning_rate=1e-3,
        weight_decay=1e-5,
        tau=0.95,
        sync_freq=1,
        **kwargs
    ):
        super().__init__()

        self.obs_shape = obs_shape
        self.tau = tau
        self.steps_until_sync = 0
        self.sync_freq = sync_freq

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # self.decoder_model = gen_model_nets.GenDecoder2D(obs_shape[0], obs_shape).cuda()
        # self.encoder = gen_model_nets.GenEncoder(obs_shape).cuda()
        self.decoder_model = gen_model_nets.GenDecoder2D(1152, obs_shape, use_grid=True).cuda()

        self.decoder_optimizer = torch.optim.Adam(
            list(self.decoder_model.parameters()),
            lr=learning_rate,
            weight_decay=self.weight_decay,
        )

        # Adam optimizer for autoencoder
        # self.optimizer = torch.optim.Adam(
        #     list(self.encoder.parameters()) + list(self.decoder_model.parameters()),
        #     lr=learning_rate,
        #     weight_decay=self.weight_decay,
        # )

		# Autoencoder AdamW
        # self.optimizer = torch.optim.AdamW(
        #     list(self.encoder.parameters()) + list(self.decoder_model.parameters()),
        #     lr=learning_rate,
        #     weight_decay=self.weight_decay,
        # )


        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, factor=0.9, patience=12)

        self.encoder = None

    def get_hyperparameters(self):
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }

    def share_dependant_models(self, model):
        self.encoder = model.encoder
        pass

    def train_step(self, batch, epoch):
        # NOTE: TEST ONLY TRAINING ONE LAYER
        obs_x = torch.as_tensor(batch["obs"], device="cuda")

        # Convert data from -1 and 1 to 0 and 1
        # obs_x = (obs_x + 1) / 2  # Now values are 0 and 1

        obs_x = obs_x[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1) # Only agent layer

        ox_encoded_online = self.encoder(obs_x).detach() # detach for running singlestep
        obs_x_reconstructed = self.decoder_model(ox_encoded_online)

        # decoder_loss = F.mse_loss(obs_x_reconstructed[:, 1, :, :], obs_x[:, 1, :, :])
        decoder_loss = F.l1_loss(obs_x_reconstructed[:, 1, :, :], obs_x[:, 1, :, :])

        # Testing loss functions
        # decoder_loss = F.binary_cross_entropy_with_logits(obs_x_reconstructed, obs_x)

        # pos_weight = torch.tensor([224]).to("cuda") # Calculate weights
        # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) # Use BCEWithLogitsLoss with pos_weight
        # decoder_loss = criterion(obs_x_reconstructed, obs_x)

        self.decoder_optimizer.zero_grad()
        decoder_loss.backward()
        self.decoder_optimizer.step()

        log = {
            "decoder_loss": decoder_loss.detach().item(),
        }
        return log

    def weighted_train_step(self, batch, epoch):
        obs_x = torch.as_tensor(batch["obs"], device="cuda")

        # Forward pass through encoder and decoder
        ox_encoded_online = self.encoder(obs_x)
        obs_x_reconstructed = self.decoder_model(ox_encoded_online)

        # Separate out the three layers
        obs_x_obstacle = obs_x[:, 0, :, :]  # Obstacle layer
        obs_x_agent = obs_x[:, 1, :, :]     # Agent position layer
        obs_x_goal = obs_x[:, 2, :, :]      # Goal position layer

        obs_x_reconstructed_obstacle = obs_x_reconstructed[:, 0, :, :]
        obs_x_reconstructed_agent = obs_x_reconstructed[:, 1, :, :]
        obs_x_reconstructed_goal = obs_x_reconstructed[:, 2, :, :]

        # Calculate the number of obstacles (1s in the obstacle layer)
        num_obstacles = torch.sum(obs_x_obstacle == 1).item()  # Count number of 1's (obstacles)

        # Ensure there are at least some obstacles to avoid division by zero
        if num_obstacles == 0:
            num_obstacles = 1  # Default to 1 obstacle if none are present, just for safety

        # Compute the losses
        obstacle_loss = F.mse_loss(obs_x_reconstructed_obstacle, obs_x_obstacle)
        agent_loss = F.mse_loss(obs_x_reconstructed_agent, obs_x_agent)
        goal_loss = F.mse_loss(obs_x_reconstructed_goal, obs_x_goal)

        # Calculate dynamic weight for obstacle layer based on the number of obstacles
        obstacle_weight = 1 / num_obstacles  # Inverse of number of obstacles
        agent_weight = 1.0  # Since there's only 1 agent
        goal_weight = 1.0   # Since there's only 1 goal

        # Compute weighted loss
        weighted_loss = (obstacle_weight * obstacle_loss) + (agent_weight * agent_loss) + (goal_weight * goal_loss)

        # Backpropagation
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()

        log = {
            "total_loss": weighted_loss.detach().item(),
            "obstacle_loss": obstacle_loss.detach().item(),
            "agent_loss": agent_loss.detach().item(),
            "goal_loss": goal_loss.detach().item(),
            "num_obstacles": num_obstacles,
            "obstacle_weight": obstacle_weight,
        }
        return log

    def normalized_train_step(self, batch, epoch):
        if self.encoder is None:
            raise ValueError("Encoder not shared. Call share_dependant_models() before training.")

        obs_x = torch.as_tensor(batch["obs"], device="cuda")

        obs_x_norm = self._normalize_input(obs_x)

        # ox_encoded_online = self.encoder(obs_x).detach()
        ox_encoded_online = self.encoder(obs_x_norm)
        obs_x_reconstructed = self.decoder_model(ox_encoded_online)

        decoder_loss = F.mse_loss(obs_x_reconstructed, obs_x_norm) # change so proportion from all channels

        # self.decoder_optimizer.zero_grad()
        # decoder_loss.backward()
        # self.decoder_optimizer.step()

        self.optimizer.zero_grad()
        decoder_loss.backward()
        self.optimizer.step()

        log = {
            "decoder_loss": decoder_loss.detach().item(),
            # "learning_rate": learning_rate,
        }
        return log

    def _normalize_input(self, obs_x):
        """Normalize the input across all channels (layers) independently."""
        obs_x_norm = obs_x.clone()
        # Normalizing each layer:
        # Layer 1: Obstacle layer (assume binary or bounded values)
        obs_x_norm[:, 0, :, :] = (obs_x[:, 0, :, :] - obs_x[:, 0, :, :].min()) / (obs_x[:, 0, :, :].max() - obs_x[:, 0, :, :].min())

        # Layer 2: Agent position layer (single point, might not need normalization if binary)
        obs_x_norm[:, 1, :, :] = obs_x[:, 1, :, :]  # Assuming 0s and 1s (no need to scale)

        # Layer 3: Goal position (fixed location, potentially no need for normalization)
        obs_x_norm[:, 2, :, :] = obs_x[:, 2, :, :]  # Assuming it's already normalized (fixed center)

        return obs_x_norm