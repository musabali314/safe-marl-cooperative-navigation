# Safe MARL Cooperative Navigation (PyTorch Reimplementation)

> **Unofficial, from-scratch reimplementation and extension of**
> Dawood et al., *Safe Multi-Agent Reinforcement Learning for Behavior-Based Cooperative Navigation* (2025).

<p align="center">

Stage 1:

https://github.com/user-attachments/assets/0f0d17f4-e8bd-4760-8f1e-0eae3a4420d5

Stage 2:

https://github.com/user-attachments/assets/c12dc15e-f163-4d96-bc1d-74c813279bf4

Stage 3:

https://github.com/user-attachments/assets/7efa9062-9660-4ddd-9822-1347de5f18e5

-->

▶️ **Stage 1 — Goal Reaching**
▶️ **Stage 2 — Formation-Keeping Cooperation**
▶️ **Stage 3 — MPC-Safe Obstacle-Aware Navigation**

</p>

---


---

## 1. Overview

This repository implements a fully self-contained Python simulation and learning stack for **safe cooperative navigation** with three differential-drive robots in a continuous 2D workspace. The implementation closely follows Dawood et al. (2025) while replacing ROS/Gazebo with a pure Python environment, integrating a centralized attention critic, and enforcing safety via an MPC-style action filter.

---

## 2. Environment and Dynamics (`env_coop.py`)

All simulation logic is implemented from scratch inside the `EnvCoop` class.

### 2.1 Workspace and Robot Model

**Workspace**

```math
Ω = [-3, 3] × [-3, 3] ⊂ ℝ²
```

**Robot State (unicycle model)**

```math
x_i = (x_i, y_i, θ_i)
```

**Control Input**

```math
a_i = (v_i, ω_i)
```

**Discrete-Time Dynamics**

```math
x_i(t+1) = x_i(t) + v_i(t) cos(θ_i(t)) Δt
y_i(t+1) = y_i(t) + v_i(t) sin(θ_i(t)) Δt
θ_i(t+1) = θ_i(t) + ω_i(t) Δt
```

Actions are bounded and consistent with the actor network.

Episodes terminate upon collision, prolonged stagnation, or reaching the step horizon.

---

### 2.2 Geometry and Obstacles

- Robots and obstacles are modeled as circles.
- Obstacles are enabled only in **Stage 3**.
- The goal is a circular region sampled to avoid collisions.

---

### 2.3 LiDAR Observation Model

Each robot is equipped with a simulated planar LiDAR.

```math
scan_i = [r_1, r_2, …, r_K],   r_k ∈ [0, r_max]
```

Rays are cast in the robot frame over `[-π, π]`. The minimum distance ray and its bearing are explicitly encoded in the observation.

---

### 2.4 Agent Observation Vector

Each agent receives a **12-dimensional local observation**:

```math
o_i =
[
d_min,
α_obs,
d_{i1}, α_{i1},
d_{i2}, α_{i2},
d_i^goal, α_i^goal,
v_i(t-1), ω_i(t-1),
d_centroid^goal,
d_form
]
```

Where:
- `d_min, α_obs` encode the closest LiDAR hit,
- `(d_{i1}, α_{i1}), (d_{i2}, α_{i2})` are distances/bearings to neighbors,
- `(d_i^goal, α_i^goal)` encode goal geometry,
- `(v_i(t-1), ω_i(t-1))` are previous actions,
- `d_centroid^goal` is team-level progress,
- `d_form` is the desired formation spacing.

All quantities are normalized.

---

### 2.5 Reward Function

The per-agent reward is composed as:

```math
r_i =
r_alive +
r_prog +
r_goal +
r_form +
r_rr +
r_obs +
r_mpc
```

**Components**

```math
r_prog ∝ d_goal(t−1) − d_goal(t)
```

```math
r_form ∝ −|d_ij − d_form|
```

```math
r_rr < 0  if  d_ij < d_safe
```

```math
r_obs < 0  if  min(scan_i) < d_safe_obs
```

```math
r_mpc ∝ −||a_RL − a_safe||
```

Terminal rewards override shaping:
- Success yields a large positive reward.
- Collisions yield a large negative reward.

---

## 3. Safety Filtering (`mpc_filter.py`)

A supervisory safety layer computes:

```math
a_safe = π_safe(a_RL, x)
```

Subject to:

```math
d_obs(x) ≥ d_safe_obs
d_nei(x) ≥ d_safe_nei
```

Two backends are supported:

### 3.1 ACADOS MPC (Exact)

Solves a constrained nonlinear optimal control problem minimizing deviation from the RL action while enforcing safety constraints over a finite horizon.

### 3.2 Sampling-Based MPC (Fallback)

Approximates MPC by:
- Sampling candidate actions near `a_RL`,
- Forward simulating trajectories,
- Penalizing unsafe predictions,
- Selecting the lowest-cost action.

This backend is used automatically if ACADOS is unavailable.

---

## 4. Multi-Agent SAC with Attention Critic

### 4.1 Actor (Shared Gaussian Policy)

Each agent follows a Gaussian policy:

```math
π_θ(a_i | o_i) = 𝒩(μ_θ(o_i), σ_θ(o_i))
```

Sampling uses reparameterization:

```math
a = tanh(μ + σ ⊙ ε)
```

The same actor network is shared across agents.

---

### 4.2 Centralized Attention Critic

The critic encodes each agent into an embedding `h_i` and applies multi-head self-attention:

```math
q_i = W_Q h_i
k_i = W_K h_i
v_i = W_V h_i
```

```math
α_ij = softmax( (q_i · k_j) / √d )
```

```math
c_i = Σ_j α_ij v_j
```

Final critic input:

```math
z_i = [h_i , c_i]
```

Twin Q-networks compute:

```math
Q_1(z), Q_2(z)
```

---

### 4.3 SAC Updates

Actor objective:

```math
J_π = E[ α log π(a|o) − Q(o,a) ]
```

Critic target:

```math
y = r + γ ( min(Q′) − α log π )
```

Temperature is learned automatically.

---

## 5. Curriculum Learning

Training proceeds in three stages:

1. **Stage 1**: Goal reaching (no formation, no obstacles, no safety filter)
2. **Stage 2**: Formation keeping (formation enabled, no obstacles)
3. **Stage 3**: Obstacle-aware navigation with MPC safety filter

Each stage builds on the previous one.

---

## 6. Training and Evaluation

- Training: `python main.py`
- Quantitative evaluation: `python Evaluation/eval_stats.py`
- Visual evaluation: `python Evaluation/eval_visual.py`
- Video recording: `python Evaluation/eval_video.py`

Evaluation outputs JSON files for each stage.

---

## 7. Repository Structure

```
safe-marl-cooperative-navigation/
├── ATT_modules.py
├── SAC_ATT.py
├── env_coop.py
├── mpc_filter.py
├── respawnGoal.py
├── utilis.py
├── main.py
├── Models/
├── Evaluation/
│   ├── eval_stats.py
│   ├── eval_visual.py
│   ├── stage*_paper_eval.json
└── README.md
```

---

## 8. Dependencies

- Python 3.10+
- numpy, torch, matplotlib, tensorboard
- Optional: acados + casadi (exact MPC)

---

## 9. Citation

```bibtex
@misc{dawood2025safemultiagentreinforcementlearning,
  title={Safe Multi-Agent Reinforcement Learning for Behavior-Based Cooperative Navigation},
  author={Murad Dawood and Sicong Pan and Nils Dengler and Siqi Zhou and Angela P. Schoellig and Maren Bennewitz},
  year={2025},
  eprint={2312.12861},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2312.12861},
}
```

---

**Research / educational use only.**
