# Safe MARL Cooperative Navigation (PyTorch Reimplementation)

> **Unofficial, from-scratch reimplementation and extension of**
> Dawood et al., *Safe Multi-Agent Reinforcement Learning for Behavior-Based Cooperative Navigation* (2025).

<p align="center">

<!--
Stage 1:
Stage 2:
Stage 3:
-->

▶️ **Stage 1 — Goal Reaching**
▶️ **Stage 2 — Formation-Keeping Cooperation**
▶️ **Stage 3 — MPC-Safe Obstacle-Aware Navigation**

</p>

---

## 1. Overview

This repository implements a fully self-contained Python simulation and learning stack for **safe cooperative navigation** with three differential-drive robots in a continuous 2D workspace. The project closely follows the ideas in Dawood et al. (2025) while:

- Replacing ROS/Gazebo with a **pure Python environment** (`env_coop.py`)
- Using a **shared Gaussian actor** and a **centralized attention critic** (`SAC_ATT.py`, `ATT_modules.py`)
- Integrating a **hybrid safety filter** (`mpc_filter.py`) that uses:
  - ACADOS-based MPC when available
  - A sampling-based approximate MPC fallback otherwise
- Providing clear **reward decomposition**, **curriculum learning**, and **evaluation scripts**

The resulting agents learn to reach a goal, maintain approximate formation, avoid collisions, and cooperate under a safety supervisor.

---

## 2. Environment and Dynamics (`env_coop.py`)

The environment is implemented from scratch and everything is controlled directly through the `EnvCoop` class.

### 2.1 Workspace and Dynamics

- **Workspace**: square domain Ω = [-3, 3] × [-3, 3] (6 × 6 meters)
- **Robots**: up to 3 unicycle robots with state

  x_i = (x_i, y_i, θ_i) ∈ R³

  and continuous control

  a_i = (v_i, ω_i) ∈ R²

- **Discrete-time unicycle model** with step size `DT = 0.1`:

  x_i(t+1) = x_i(t) + v_i(t) cos(θ_i(t)) Δt
  y_i(t+1) = y_i(t) + v_i(t) sin(θ_i(t)) Δt
  θ_i(t+1) = θ_i(t) + ω_i(t) Δt

- **Velocity limits** (shared with the actor):
  - Linear: v_i ∈ [-0.5, 0.5]
  - Angular: ω_i ∈ [-0.5, 0.5]

Episodes terminate when:
- Any robot collides with an obstacle or another robot
- A robot is stuck too long (no significant motion)
- Or the maximum step horizon `MAX_STEPS = 350` is reached

### 2.2 Geometry, Obstacles, and Goal

- **Robot radius**: `ROBOT_RADIUS = 0.15`
- **Obstacle radius**: `OBSTACLE_RADIUS = 0.25`
- **Number of obstacles**: `N_OBSTACLES = 3` (Stage 3 only)
- Obstacles are sampled with a clearance from robot initial positions
- The **goal** is a circular region managed by `RespawnGoal`, resampled after each success

### 2.3 LiDAR Model

Each robot carries a simulated 1D LiDAR:

- **Number of rays**: `LIDAR_RAYS = 40`
- **Range**: `LIDAR_RANGE = 2.5` meters
- Rays are cast in the robot frame over [-π, π]

For each ray:
- Intersection with walls is checked
- Intersection with obstacles is checked
- The minimum hit distance is returned; if nothing is hit, distance = `LIDAR_RANGE`

Additionally, for each robot the closest obstacle point in world coordinates is stored and passed to the safety filter.

### 2.4 Observations

Each agent receives a **12-dimensional local observation**:

o_i = [
  min_scan,
  angle_closest_obs,
  d_i1, angle_i1,
  d_i2, angle_i2,
  d_i_goal, angle_i_goal,
  v_i(t-1), ω_i(t-1),
  d_centroid_goal, d_form_ref
]

This encodes:
- Nearest obstacle distance and bearing
- Distances and bearings to the two closest robots
- Agent-to-goal distance and heading
- Previous action (for smoothness)
- Team centroid-to-goal distance
- Desired formation spacing (sampled from {1.0, 1.25, 1.5})

All quantities are normalized by workspace size or sensor limits.

### 2.5 Reward Structure

The per-agent reward is:

r = r_alive + r_prog + r_goal + r_form + r_rr + r_obs + r_mpc

Where:
- **Alive reward**: small positive constant
- **Progress reward**: proportional to decrease in centroid-to-goal distance
- **Goal shaping**: penalty proportional to remaining goal distance
- **Formation penalty**: active in Stages 2–3, penalizes deviation from desired spacing
- **Robot–robot penalty**: soft + hard penalties for close proximity
- **Obstacle penalty**: active in Stage 3, penalizes small LiDAR distances
- **MPC deviation penalty**: penalizes deviation between RL and safety-filtered actions

Terminal overrides:
- **Goal reached**: large positive reward, goal respawned
- **Collision or stagnation**: large negative reward

A `reward_info` dictionary is returned for logging and debugging.

---

## 3. Safety Filtering (`mpc_filter.py`)

A supervisory safety filter modifies the RL action u_RL into a safe action u_safe subject to:

- Robot–obstacle distance ≥ d_safe_obs
- Robot–robot distance ≥ d_safe_nei

Two backends are implemented:

1. **ACADOS-based MPC**
   - Solves a short-horizon nonlinear OCP with unicycle dynamics
   - Objective: minimize deviation from u_RL while enforcing safety constraints

2. **Sampling-based MPC fallback**
   - Used when ACADOS / CasADi are unavailable
   - Evaluates a grid of candidate actions around u_RL
   - Simulates short-horizon trajectories
   - Penalizes predicted constraint violations

Unified API:

```python
safe_action = filter_mpc().run_mpc(
    u_RL, x0,
    obs_x, obs_y,
    n1_x, n1_y,
    n2_x, n2_y,
    agent_index
)
```

---

## 4. Multi-Agent SAC with Attention Critic

### 4.1 Actor

- Shared Gaussian policy π(a | o)
- Conv1D + MLP LiDAR encoder
- MLP observation encoder
- Outputs mean μ and std σ
- Actions sampled via reparameterization and squashed with `tanh`

### 4.2 Centralized Attention Critic

- Per-agent embeddings h_i
- Multi-head self-attention over agents
- Context vectors c_i capture inter-agent influence
- Twin Q-networks reduce overestimation bias

### 4.3 SAC Updates

- Actor loss: α log π(a|o) − Q(o,a)
- Critic loss: MSE against soft Bellman target
- Temperature α learned automatically toward target entropy

---

## 5. Curriculum Learning

Stages are configured via `create_env_for_stage(stage)`:

1. **Stage 1 — Goal Reaching**
   - No obstacles, no formation, no safety filter

2. **Stage 2 — Formation-Keeping**
   - Formation enabled
   - No obstacles, no safety filter

3. **Stage 3 — MPC-Safe Navigation**
   - Obstacles enabled
   - Formation enabled
   - MPC safety filter enabled

---

## 6. Training and Evaluation

### Training
```bash
python main.py
```

### Evaluation
```bash
python eval_stats.py
```

### Visualization
- `eval_visual.py` for live plots
- `eval_video.py` for MP4 recordings

---

## 7. Repository Structure

```text
.
├── ATT_modules.py
├── SAC_ATT.py
├── env_coop.py
├── mpc_filter.py
├── respawnGoal.py
├── utilis.py
├── main.py
├── Evaluation/
│   ├── eval_stats.py
│   ├── eval_visual.py
│   ├── stage*_paper_eval.json
├── Models/
│   └── 5000 episodes/
└── README.md
```

---

## 8. Dependencies

- Python 3.10+
- numpy
- torch
- matplotlib
- tensorboard
- (Optional) acados, casadi

---

## 9. Citation

If you use this work, cite the original paper:

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
