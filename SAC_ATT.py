#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from ATT_modules import CentralAttention1
from utilis import soft_update

# physical limits (match environment)
ACTION_V_MAX = 0.5
ACTION_W_MAX = 0.5


# ============================================================================
#                              ACTOR NETWORK
# ============================================================================
class SACActor(nn.Module):
    """
    Shared decentralized Gaussian policy π_θ(a_i | o_i).
    Implements:

    • Observation encoder  φ(o_i) → h_i  (learned nonlinear embedding)
    • Gaussian policy heads μ(h_i), σ(h_i)
    • Reparameterization:  z = μ + σ ⊙ ε,  ε ~ N(0, I)
    • Squashing:           tanh(z) → bounded actions
    """

    def __init__(self, state_dim, action_dim, scan_size,
                 log_std_min=-10.0, log_std_max=2.0):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.scan_size = scan_size

        self.LOG_STD_MIN = log_std_min
        self.LOG_STD_MAX = log_std_max

        # ---------------------------------------------------------
        # 1) LIDAR FEATURE EXTRACTOR φ_scan
        #    Implements φ_scan(s) = MLP(Conv1D(s))
        # ---------------------------------------------------------
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.Flatten()
        )

        # determine conv output dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 1, scan_size)
            conv_out = self.conv(dummy).shape[1]

        # final LiDAR embedding:  φ_scan ∈ ℝ¹⁰
        self.scan_fc = nn.Sequential(
            nn.Linear(conv_out, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

        # ---------------------------------------------------------
        # 2) GENERAL OBSERVATION ENCODER φ(o_i)
        #    h_i = σ(W₂ σ(W₁ [o_i , φ_scan] + b₁) + b₂)
        # ---------------------------------------------------------
        self.fc = nn.Sequential(
            nn.Linear(state_dim + 10, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # ---------------------------------------------------------
        # 3) Gaussian policy heads:
        #       μ(h_i)  and  log σ(h_i)
        #    These define  π_θ(a_i | o_i) = N(μ, σ² I)
        # ---------------------------------------------------------
        self.mu_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    # ---------------------------------------------------------
    # Computes latent embedding h_i = φ(o_i)
    # ---------------------------------------------------------
    def _encode(self, state, scan):
        scan = scan.unsqueeze(1)
        scan_feat = self.scan_fc(self.conv(scan))
        h = self.fc(torch.cat([state, scan_feat], dim=-1))
        return h

    # ---------------------------------------------------------
    # Forward pass produces (μ_i, σ_i) for Gaussian policy
    # ---------------------------------------------------------
    def forward(self, state, scan):
        h = self._encode(state, scan)
        mu = self.mu_head(h)

        # bound log σ  → guarantees numerical stability
        log_std = torch.clamp(self.log_std_head(h),
                              self.LOG_STD_MIN, self.LOG_STD_MAX)

        return mu, torch.exp(log_std)

    # ---------------------------------------------------------
    # Sample action using reparameterization:
    #
    #   z = μ + σ ε , ε ~ N(0, 1)
    #   a = tanh(z) → bounded continuous control
    #
    # Includes SAC log-probability correction for the tanh transform.
    # ---------------------------------------------------------
    def sample(self, state, scan):
        mu, std = self.forward(state, scan)

        dist = Normal(mu, std)
        z = dist.rsample()        # differentiable sample
        u = torch.tanh(z)         # squash to [-1, 1]

        # scale to physical limits
        a = torch.cat([
            u[:, :1] * ACTION_V_MAX,
            u[:, 1:2] * ACTION_W_MAX
        ], dim=-1)

        # SAC log-prob correction:
        # log π(a) = log N(z) - Σ log(1 - tanh(z)^2)
        log_pi = dist.log_prob(z).sum(-1, keepdim=True)
        log_pi -= (2 * (np.log(2) - z - F.softplus(-2 * z))).sum(-1, keepdim=True)

        return a, log_pi, torch.tanh(mu)

    # deterministic evaluation policy (mu only)
    def act_deterministic(self, state, scan):
        mu, _ = self.forward(state, scan)
        u = torch.tanh(mu)
        return torch.cat([
            u[:, :1] * ACTION_V_MAX,
            u[:, 1:2] * ACTION_W_MAX
        ], dim=-1)


# ============================================================================
#                               SAC ALGORITHM
# ============================================================================
class SAC:
    """
    Implements the full Soft Actor–Critic update pipeline:

    • π_θ  (Gaussian actor, shared across agents)
    • Q_φ  (centralized attention critic over 3 agents)
    • Target critic  φ'
    • Temperature α for entropy regularization

    Mathematical updates implemented:

    Actor:
        J_π = E[ α log π(a|o) - Q(o,a) ]

    Critic:
        y = r + γ ( min(Q_φ'(s',a')) - α log π(a'|o') )
        J_Q = E[ (Q - y)² ]

    Temperature:
        J_α = E[ -α ( log π(a|o) + H_target ) ]
    """

    def __init__(self,
                 seed,
                 state_dim,
                 actor_state_dim,
                 action_dim,
                 max_action_v,
                 max_action_w,
                 scan_size,
                 replay_buffer,
                 discount=0.99,
                 reward_scale=2.0,
                 batch_size=256):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # seeding for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.batch_size = batch_size
        self.discount = discount
        self.reward_scale = reward_scale

        self.state_dim = state_dim
        self.actor_state_dim = actor_state_dim
        self.action_dim = action_dim
        self.scan_size = scan_size

        # fixed team size for critic
        self.n_agents = 3

        # sync action limits
        global ACTION_V_MAX, ACTION_W_MAX
        ACTION_V_MAX = max_action_v
        ACTION_W_MAX = max_action_w

        # replay buffer
        self.replay_buffer = replay_buffer

        # ---------------------------------------------------------
        # Shared decentralized Gaussian policy  π_θ
        # ---------------------------------------------------------
        self.actor = SACActor(actor_state_dim, action_dim, scan_size).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        # ---------------------------------------------------------
        # Centralized attention critic Q_φ(o,a)
        # Implements multi-head attention over agents.
        # ---------------------------------------------------------
        self.critic = CentralAttention1(
            state_size=actor_state_dim,
            action_size=action_dim,
            no_agents=self.n_agents,
            scan_size=scan_size,
            hidden_dim=128,
            attend_heads=4
        ).to(self.device)

        self.critic_target = CentralAttention1(
            state_size=actor_state_dim,
            action_size=action_dim,
            no_agents=self.n_agents,
            scan_size=scan_size,
            hidden_dim=128,
            attend_heads=4
        ).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

        # ---------------------------------------------------------
        # Entropy temperature α (learned)
        # ---------------------------------------------------------
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -float(action_dim)

    # ============================================================================
    # Action selection  π_θ(a|o)
    # ============================================================================
    def select_action(self, actor_states, scans, eval=False):

        s = torch.tensor(actor_states, device=self.device, dtype=torch.float32)
        sc = torch.tensor(scans, device=self.device, dtype=torch.float32)

        self.actor.eval()
        with torch.no_grad():
            if eval:
                a = self.actor.act_deterministic(s, sc)
            else:
                a, _, _ = self.actor.sample(s, sc)

        return a.cpu().numpy()

    # ============================================================================
    # Main SAC Training Update
    # ============================================================================
    def train(self):

        if self.replay_buffer.len() < self.batch_size:
            return 0.0, 0.0, self.log_alpha.exp().item()

        (state_critic, next_state_critic, reward, not_done,
         actor_state, actor_state_next, action, scan, scan_next) = \
            self.replay_buffer.sample(self.batch_size)

        B = self.batch_size
        N = self.n_agents
        BN = B * N

        # reshape all agent data → flat batch for critic/actor
        s   = actor_state.view(BN, -1).to(self.device)
        sn  = actor_state_next.view(BN, -1).to(self.device)
        a   = action.view(BN, -1).to(self.device)
        sc  = scan.view(BN, -1).to(self.device)
        scn = scan_next.view(BN, -1).to(self.device)

        r  = reward.view(BN, 1).to(self.device)
        nd = not_done.view(BN, 1).to(self.device)

        # ===============================
        # 1) ACTOR UPDATE  (J_π)
        # ===============================
        a_new, log_pi, _ = self.actor.sample(s, sc)

        alpha = torch.clamp(self.log_alpha.exp(), 1e-5, 10.0)

        q1_pi, q2_pi = self.critic(s, a_new, sc)
        min_q = torch.min(q1_pi, q2_pi)

        actor_loss = (alpha * log_pi - min_q).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()

        # ===============================
        # 2) TEMPERATURE α UPDATE
        # ===============================
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.log_alpha], 1.0)
        self.alpha_optim.step()

        with torch.no_grad():
            self.log_alpha.clamp_(min=-10.0, max=2.0)

        # ===============================
        # 3) CRITIC UPDATE  (Bellman backup)
        # ===============================
        with torch.no_grad():
            a_next, log_pi_next, _ = self.actor.sample(sn, scn)
            q1_t, q2_t = self.critic_target(sn, a_next, scn)
            min_q_t = torch.min(q1_t, q2_t)

            q_target = r * self.reward_scale + \
                       nd * self.discount * (min_q_t - alpha * log_pi_next)

            q_target = torch.clamp(q_target, -20.0, 20.0)

        q1, q2 = self.critic(s, a, sc)

        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()

        soft_update(self.critic_target, self.critic, tau=1e-3)

        return actor_loss.item(), critic_loss.item(), alpha.item()


    # ============================================================================
    # Save / Load
    # ============================================================================
    def save(self, prefix):
        torch.save(self.actor.state_dict(),  prefix + "_actor.pth")
        torch.save(self.critic.state_dict(), prefix + "_critic.pth")
        torch.save(self.critic_target.state_dict(), prefix + "_critic_target.pth")
        torch.save(self.log_alpha,            prefix + "_log_alpha.pth")

    def load(self, prefix):
        self.actor.load_state_dict(torch.load(prefix + "_actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(prefix + "_critic.pth", map_location=self.device))
        self.critic_target.load_state_dict(torch.load(prefix + "_critic_target.pth", map_location=self.device))
        self.log_alpha = torch.load(prefix + "_log_alpha.pth", map_location=self.device)
