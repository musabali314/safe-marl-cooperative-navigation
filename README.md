# Safe MARL Cooperative Navigation (PyTorch Reimplementation)

> **Unofficial, from-scratch reimplementation and extension of**
> Dawood et al., *Safe Multi-Agent Reinforcement Learning for Behavior-Based Cooperative Navigation* (2025).

<p align="center">

Stage 1:

https://github.com/user-attachments/assets/b42c546a-bfd1-4ede-a9e9-cbbb1a269f04

Stage 2:

https://github.com/user-attachments/assets/488fc954-7267-40aa-be8b-3bc06d46241a

Stage 3:

https://github.com/user-attachments/assets/ed1134a8-a1ce-45ec-bec6-8e1741242e7d

</p>

▶️ **Stage 1 — Goal Reaching**
▶️ **Stage 2 — Formation-Keeping Cooperation**
▶️ **Stage 3 — MPC-Safe Obstacle-Aware Navigation**

---

## 1. Overview

This repository implements a fully self-contained Python simulation and learning stack for **safe cooperative navigation** with three differential-drive robots in a continuous 2D workspace. The implementation closely follows Dawood et al. (2025) while replacing ROS/Gazebo with a pure Python environment, integrating a centralized attention critic, and enforcing safety via an MPC-style action filter.

---

## 2. Environment and Dynamics (`env_coop.py`)

All simulation logic is implemented from scratch inside the `EnvCoop` class.

### 2.1 Workspace and Robot Model

**Workspace**

$$\Omega = [-3, 3] \times [-3, 3] \subset \mathbb{R}^2$$

**Robot State (unicycle model)**

$$\mathbf{x}_i = (x_i, y_i, \theta_i)$$

**Control Input**

$$\mathbf{a}_i = (v_i, \omega_i)$$

**Discrete-Time Dynamics**

$$
\begin{align}
x_i(t+1) &= x_i(t) + v_i(t) \cdot \cos(\theta_i(t)) \cdot \Delta t \\
y_i(t+1) &= y_i(t) + v_i(t) \cdot \sin(\theta_i(t)) \cdot \Delta t \\
\theta_i(t+1) &= \theta_i(t) + \omega_i(t) \cdot \Delta t
\end{align}
$$

Actions are bounded and consistent with the actor network. Episodes terminate upon collision, prolonged stagnation, or reaching the step horizon.

---

### 2.2 Geometry and Obstacles

- Robots and obstacles are modeled as circles.
- Obstacles are enabled only in **Stage 3**.
- The goal is a circular region sampled to avoid collisions.

---

### 2.3 LiDAR Observation Model

Each robot is equipped with a simulated planar LiDAR.

$$\text{scan}_i = [r_1, r_2, \ldots, r_K], \quad r_k \in [0, r_{\text{max}}]$$

Rays are cast in the robot frame over $[-\pi, \pi]$. The minimum-distance ray and its bearing are explicitly encoded in the observation.

---

### 2.4 Agent Observation Vector

Each agent receives a **12-dimensional local observation**:

$$
\mathbf{o}_i = \begin{bmatrix}
d_{\text{min}} \\
\alpha_{\text{obs}} \\
d_{i1}, & \alpha_{i1} \\
d_{i2}, & \alpha_{i2} \\
d_{i,\text{goal}}, & \alpha_{i,\text{goal}} \\
v_i(t-1), & \omega_i(t-1) \\
d_{\text{centroid,goal}} \\
d_{\text{form,ref}}
\end{bmatrix}
$$

All quantities are normalized.

---

### 2.5 Reward Function

The per-agent reward is composed as:

$$
r_i = r_{\text{alive}} + r_{\text{prog}} + r_{\text{goal}} + r_{\text{form}} + r_{\text{rr}} + r_{\text{obs}} + r_{\text{mpc}}
$$

Key terms:

$$
\begin{align}
r_{\text{prog}} &\sim d_{\text{goal}}(t-1) - d_{\text{goal}}(t) \\
r_{\text{form}} &\sim -|d_{ij} - d_{\text{form,ref}}| \\
r_{\text{rr}} &< 0 \quad \text{if } d_{ij} < d_{\text{safe}} \\
r_{\text{obs}} &< 0 \quad \text{if } \min(\text{scan}_i) < d_{\text{safe,obs}} \\
r_{\text{mpc}} &\sim -\|\mathbf{a}_{\text{RL}} - \mathbf{a}_{\text{safe}}\|
\end{align}
$$

Terminal rewards override shaping:

- Goal reached → large positive reward
- Collision or stagnation → large negative reward

---

## 3. Safety Filtering (`mpc_filter.py`)

A supervisory safety layer computes:

$$\mathbf{a}_{\text{safe}} = \pi_{\text{safe}}(\mathbf{a}_{\text{RL}}, \mathbf{x})$$

Subject to distance constraints:

$$
\begin{align}
d_{\text{obs}}(\mathbf{x}) &\geq d_{\text{safe,obs}} \\
d_{\text{nei}}(\mathbf{x}) &\geq d_{\text{safe,nei}}
\end{align}
$$

Two backends are supported:

### 3.1 ACADOS MPC (Exact)

Solves a constrained nonlinear optimal control problem minimizing deviation from the RL action while enforcing safety constraints over a finite horizon.

### 3.2 Sampling-Based MPC (Fallback)

- Samples candidate actions near $\mathbf{a}_{\text{RL}}$
- Forward-simulates trajectories
- Penalizes unsafe predictions
- Selects the lowest-cost action

Used automatically if ACADOS is unavailable.

---

## 4. Multi-Agent SAC with Attention Critic

### 4.1 Actor (Shared Gaussian Policy)

Each agent follows a Gaussian policy:

$$\pi_\theta(\mathbf{a}_i \mid \mathbf{o}_i) = \mathcal{N}(\mu_\theta(\mathbf{o}_i), \sigma_\theta(\mathbf{o}_i))$$

Sampling uses reparameterization:

$$\mathbf{a} = \tanh(\mu + \sigma \odot \epsilon)$$

The same actor network is shared across agents.

---

### 4.2 Centralized Attention Critic

Attention mechanism:

$$
\begin{align}
\mathbf{q}_i &= W_Q \mathbf{h}_i \\
\mathbf{k}_i &= W_K \mathbf{h}_i \\
\mathbf{v}_i &= W_V \mathbf{h}_i
\end{align}
$$

$$\alpha_{ij} = \text{softmax}\left(\frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d}}\right)$$

$$\mathbf{c}_i = \sum_j \alpha_{ij} \mathbf{v}_j$$

Final critic embedding:

$$\mathbf{z}_i = [\mathbf{h}_i \,;\, \mathbf{c}_i]$$

Twin Q-networks compute:

$$Q_1(\mathbf{z}), \quad Q_2(\mathbf{z})$$

---

### 4.3 SAC Updates

Actor objective:

$$J_\pi = \mathbb{E}\left[\alpha \log \pi(\mathbf{a} \mid \mathbf{o}) - Q(\mathbf{o}, \mathbf{a})\right]$$

Critic target:

$$y = r + \gamma \left(\min Q_{\text{target}} - \alpha \log \pi\right)$$

Entropy temperature is learned automatically.

---

## 5. Curriculum Learning

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

**Research / educational use only.**
