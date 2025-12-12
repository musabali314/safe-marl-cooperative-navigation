#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

HID_SIZE = 256
OUT_FEATURES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# Initialization utility
# =====================================================================
def init_layer(m):
    """
    Kaiming initialization for linear/conv layers,
    matching assumptions of ReLU-like nonlinearities.
    """
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# =====================================================================
# CENTRALIZED ATTENTION CRITIC — Implements:
#
#   • Per-agent embedding   h_i = φ(o_i, a_i, lidar_i)
#   • Multi-head attention  c_i = Σ_j α_ij v_j
#   • Final critic state    z_i = [h_i , c_i]
#   • Twin Q-functions      Q1(z), Q2(z)
#
# HARD-CODED FOR EXACTLY 3 AGENTS.
# =====================================================================
class CentralAttention1(nn.Module):
    """
    Centralized critic Q_φ(o_1,a_1,o_2,a_2,o_3,a_3)
    using multi-head attention to model inter-agent influence.

    Inputs (flattened across 3 agents):
        obs  ∈ ℝ^{B*3 × state_dim}
        acts ∈ ℝ^{B*3 × action_dim}
        scan ∈ ℝ^{B*3 × scan_size}

    Output:
        (Q1, Q2) twin soft actor–critic value estimates
    """

    def __init__(
        self,
        state_size,
        action_size,
        no_agents,      # kept for API compatibility but always = 3
        scan_size,
        hidden_dim=128,
        attend_heads=4,
        batch_size=256
    ):
        super().__init__()

        assert hidden_dim % attend_heads == 0, \
            "hidden_dim must be divisible by number of heads"

        self.state_size = state_size
        self.action_size = action_size
        self.nagents = 3
        self.scan_size = scan_size
        self.hidden_dim = hidden_dim
        self.attend_heads = attend_heads
        self.batch_size = batch_size

        # ==================================================================
        # 1) LIDAR FEATURE ENCODER  φ_lidar(scan_i)
        #
        #    Extracts geometric features from raw 1D scan.
        #    This corresponds to the nonlinear transformation:
        #         φ_scan = MLP(Conv1D(scan_i))
        #
        #    Used to represent obstacle distances/angles for critic input.
        # ==================================================================
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 5),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.ReLU(),
            nn.Flatten()
        )
        test = torch.ones((1, 1, scan_size), dtype=torch.float32)
        with torch.no_grad():
            n_features = self.conv(test).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, OUT_FEATURES)
        )

        # ==================================================================
        # 2) ENCODER FOR φ(o_i, a_i, lidar_i)
        #
        #    This constructs the per-agent embedding:
        #
        #       h_i = φ(o_i, a_i, lidar_i)
        #
        #    → captures each agent’s local viewpoint + chosen control.
        # ==================================================================
        inp_dim = state_size + action_size + OUT_FEATURES

        self.encoder = nn.Sequential(
            nn.LayerNorm(inp_dim),          # normalize joint features
            nn.Linear(inp_dim, hidden_dim),
            nn.LeakyReLU()
        )

        # ==================================================================
        # 3) ATTENTION PROJECTIONS
        #
        #    For each embedded vector h_i, compute:
        #
        #       q_i = W_Q h_i     (query)
        #       k_i = W_K h_i     (key)
        #       v_i = W_V h_i     (value)
        #
        #    Multi-head form: projections split into H heads.
        #
        #    Attention weights:
        #       α_ij = softmax( (q_i ⋅ k_j) / √d )
        #
        #    Context vector:
        #       c_i = Σ_j α_ij v_j
        #
        # ==================================================================
        self.key_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)

        # ==================================================================
        # 4) TWIN Q-FUNCTIONS
        #
        #    After attention, each agent has:
        #
        #         z_i = [ h_i , c_i ]  ∈ ℝ^{2H}
        #
        #    Flattened across agents, this forms critic input.
        #
        #    Two Q heads prevent value overestimation:
        #         Q1(z), Q2(z)
        # ==================================================================
        self.q1 = nn.Sequential(
            nn.Linear(2 * hidden_dim, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1)
        )

        self.q2 = nn.Sequential(
            nn.Linear(2 * hidden_dim, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1)
        )

        self.apply(init_layer)

    # -----------------------------------------------------------------
    def conv_forward(self, scan):
        """
        Compute φ_lidar using conv + MLP.
        Produces obstacle embedding for the critic.
        """
        x = self.conv(scan.view(scan.shape[0], 1, scan.shape[1]))
        return self.fc(x)

    # -----------------------------------------------------------------
    def forward(self, obs, acts, scan):
        """
        Full critic pipeline:

            o_i, a_i, lidar_i
                 ↓  (encoding)
                h_i = φ(o_i, a_i, lidar_i)
                 ↓  (attention)
                c_i = attention(h_1, h_2, h_3)
                 ↓
                z_i = [h_i , c_i]
                 ↓
              Q1(z), Q2(z)
        """

        B_total = obs.shape[0]
        assert B_total % self.nagents == 0
        B = B_total // self.nagents

        # =============================================================
        # Step 1: Compute φ_lidar for each agent
        # =============================================================
        lidar_feat = self.conv_forward(scan)

        # =============================================================
        # Step 2: Per-agent feature fusion → h_i
        # =============================================================
        fused = torch.cat([obs, acts, lidar_feat], dim=1)
        embed = self.encoder(fused)             # shape: [B*3, H]

        # reshape into [B, 3 agents, H]
        x = embed.view(B, self.nagents, self.hidden_dim)

        # =============================================================
        # Step 3: Multi-head attention — compute q_i, k_j, v_j
        # =============================================================
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        H = self.attend_heads
        head_dim = self.hidden_dim // H

        # reshape for heads
        Q = Q.view(B, self.nagents, H, head_dim).permute(0, 2, 1, 3)
        K = K.view(B, self.nagents, H, head_dim).permute(0, 2, 1, 3)
        V = V.view(B, self.nagents, H, head_dim).permute(0, 2, 1, 3)

        # =============================================================
        # Attention weights α_ij = softmax( (q_i · k_j) / √d )
        # =============================================================
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
        attn = attn_scores.softmax(dim=-1)

        attn = F.dropout(attn, p=0.05, training=self.training)

        # =============================================================
        # Context vector c_i = Σ_j α_ij v_j
        # =============================================================
        context = torch.matmul(attn, V)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(B, self.nagents, self.hidden_dim)

        # =============================================================
        # Step 4: Final critic embedding: z_i = [h_i , c_i]
        # =============================================================
        x_final = torch.cat([x, context], dim=2)
        x_final = F.layer_norm(x_final, x_final.shape[-1:])

        # flatten across agents
        x_flat = x_final.reshape(B_total, 2 * self.hidden_dim)

        # =============================================================
        # Step 5: Twin Q-value outputs
        # =============================================================
        q1 = self.q1(x_flat)
        q2 = self.q2(x_flat)

        # stability guard
        q1 = torch.clamp(q1, -50.0, 50.0)
        q2 = torch.clamp(q2, -50.0, 50.0)

        if torch.isnan(q1).any() or torch.isnan(q2).any():
            print("⚠ NaN detected in critic output — clamping")
            q1 = torch.nan_to_num(q1, nan=0.0)
            q2 = torch.nan_to_num(q2, nan=0.0)

        return q1, q2
