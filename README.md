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

- Replacing ROS/Gazebo with a **pure Python environment** (`env_coop.py`),
- Using a **shared Gaussian actor** and a **centralized attention critic** (`SAC_ATT.py`, `ATT_modules.py`),
- Integrating a **hybrid safety filter** (`mpc_filter.py`) that uses:
  - ACADOS-based MPC when available, or
  - A sampling-based approximate MPC fallback otherwise,
- Providing clear **reward decomposition**, **curriculum learning**, and **evaluation scripts**.

The resulting agents learn to reach a goal, maintain approximate formation, avoid collisions, and cooperate under a safety supervisor.

---

## 2. Environment and Dynamics (`env_coop.py`)

The environment is implemented from scratch and everything is controlled directly through the `EnvCoop` class.

### 2.1 Workspace and Dynamics

- **Workspace**: square domain \(\Omega = [-3, 3] \times [-3, 3] \subset \mathbb{R}^2\) (\(6\times 6\) meters).
- **Robots**: up to 3 unicycle robots with state
  \[
    x_i = (x_i, y_i, \theta_i) \in \mathbb{R}^3,
  \]
  and continuous control
  \[
    a_i = (v_i, \omega_i) \in \mathbb{R}^2.
  \]
- **Discrete-time unicycle model** with step size `DT = 0.1`:
  \[
  \begin{aligned}
    x_i^{t+1} &= x_i^t + v_i^t \cos\theta_i^t\,\Delta t, \\
    y_i^{t+1} &= y_i^t + v_i^t \sin\theta_i^t\,\Delta t, \\
    \theta_i^{t+1} &= \theta_i^t + \omega_i^t\,\Delta t.
  \end{aligned}
  \]
- **Velocity limits** (shared with the actor):
  - Linear: \(v_i \in [-0.5, 0.5]\),
  - Angular: \(\omega_i \in [-0.5, 0.5]\).

Episodes terminate when:
- Any robot collides with an obstacle or another robot,
- A robot is stuck too long (no significant motion),
- Or the maximum step horizon `MAX_STEPS = 350` is reached.

### 2.2 Geometry, Obstacles, and Goal

- **Robot radius**: `ROBOT_RADIUS = 0.15`.
- **Obstacle radius**: `OBSTACLE_RADIUS = 0.25`.
- **Number of obstacles**: `N_OBSTACLES = 3` (only enabled in Stage 3).
- Obstacles are sampled with a clearance from robot initial positions.
- The **goal** is a circular region managed by `RespawnGoal`, resampled after each success.

### 2.3 LiDAR Model

Each robot carries a simulated 1D LiDAR:

- **Number of rays**: `LIDAR_RAYS = 40`.
- **Range**: `LIDAR_RANGE = 2.5` meters.
- Rays are cast in the robot frame over \([-\pi, \pi]\). For each ray, we compute:
  - Intersection with walls,
  - Intersection with obstacles,
  - And keep the smallest distance; if nothing is hit, distance = `LIDAR_RANGE`.

Additionally, for each robot we keep the **closest obstacle point** in world coordinates. This is used by the safety filter (MPC / sampling backend) to enforce distance constraints.

### 2.4 Observations

Each agent receives a **12-dimensional local observation**:

\[
o_i = [
  \text{min\_scan},
  \alpha^{\text{closest}},
  d_{i1}, \alpha_{i1},
  d_{i2}, \alpha_{i2},
  d_i^{\text{goal}}, \alpha_i^{\text{goal}},
  v_i^{t-1}, \omega_i^{t-1},
  d_{\text{centroid-goal}}, d_{\text{form-ref}}
].
\]

Intuitively this encodes:

- Nearest obstacle distance and bearing from LiDAR,
- Distances and bearings to the two closest robots,
- Agent-to-goal distance and heading,
- Previous action (for smoothness / inertia),
- Team centroid-to-goal distance,
- Desired formation spacing selected at reset from \(\{1.0, 1.25, 1.5\}\).

Observations are normalized by workspace and sensor ranges to keep values in reasonable numeric scales.

### 2.5 Reward Structure

The per-agent reward is composed in `_compute_reward` as:

\[
r = r_{\text{alive}} + r_{\text{prog}} + r_{\text{goal}} + r_{\text{form}}
    + r_{\text{rr}} + r_{\text{obs}} + r_{\text{mpc}}.
\]

- **Alive reward**: small positive constant to encourage progress without early termination.
- **Progress term**: based on centroid-to-goal distance decrease:
  \(r_{\text{prog}} \propto d_{\text{goal}}^{t-1} - d_{\text{goal}}^t\).
- **Goal shaping**: penalty proportional to the remaining centroid distance to goal.
- **Formation penalty**: active when formation is enabled (Stage 2 and 3), penalizes deviation of pairwise distances from the sampled formation spacing.
- **Robot–robot proximity penalty**: soft and hard penalties when robots come closer than a safety band (smooth cosine penalty + strong penalty inside a hard radius).
- **Obstacle proximity penalty**: when obstacles are enabled (Stage 3), penalizes small LiDAR distances in a safety band around obstacles.
- **MPC deviation penalty**: penalizes the difference between the RL action and the safety-filtered action whenever the MPC/filter modifies the command.

Terminal events override the shaping reward:

- **Goal reached (centroid near goal)**: large positive reward, goal respawned.
- **Collision or prolonged stagnation**: large negative reward, collision counter incremented.

The environment also returns a per-step `reward_info` dictionary summarizing the contribution of each term for logging and analysis.

---

## 3. Safety Filtering (`mpc_filter.py`)

Safety is enforced by a **supervisory filter** that takes the RL action \(u_{\text{RL}}\) and produces a corrected action \(u_{\text{safe}}\) that respects distance constraints:

- **Robot–obstacle distance** \(d_{\text{obs}} \geq d_{\text{safe,obs}}\),
- **Robot–robot distance** \(d_{\text{nei}} \geq d_{\text{safe,nei}}\).

Two backends are available:

1. **ACADOS-based MPC** (`_AcadosFilter`):
   - Solves a short-horizon nonlinear optimal control problem with unicycle dynamics and hard distance constraints.
   - Objective: keep \(u\) close to \(u_{\text{RL}}\) while remaining safe.

2. **Sampling-based MPC approximation** (`_SamplingFallback`):
   - Used automatically when ACADOS or `casadi` are not available.
   - Builds a discrete grid of candidate actions around \(u_{\text{RL}}\),
   - Simulates a short horizon for each candidate,
   - Penalizes predicted constraint violations and distance inverses,
   - Returns the candidate with minimal cost (plus a large penalty if any safety constraints would be violated).

The wrapper class `filter_mpc` decides at runtime which backend to use and exposes a unified API:

```python
safe_action = filter_mpc().run_mpc(
    u_RL, x0,
    obs_x, obs_y,
    n1_x, n1_y,
    n2_x, n2_y,
    agent_index
)
```

When running on a machine **without ACADOS**, the code automatically prints a message and falls back to the sampling-based safety layer. Evaluations are still meaningful, but the safety filter is approximate rather than solving the exact MPC problem.

---

## 4. Multi-Agent SAC with Attention Critic (`SAC_ATT.py`, `ATT_modules.py`)

### 4.1 Actor: Shared Gaussian Policy

The actor (`SACActor`) implements a **shared decentralized Gaussian policy** \(\pi_\theta(a_i \mid o_i)\):

1. **LiDAR encoder** (Conv1D + MLP) compresses the 40-beam scan into a 10D feature vector.
2. **Observation encoder** fuses the 12D state and 10D LiDAR embedding into a 256D latent vector.
3. Two linear heads output the Gaussian parameters:
   - Mean \(\mu_\theta(h_i)\),
   - Log standard deviation \(\log \sigma_\theta(h_i)\), clamped for numerical stability.
4. Actions are sampled by reparameterization (for training), squashed with `tanh`, and scaled to \([-0.5, 0.5]\) in both linear and angular velocity.

The same actor network is shared across all agents, enforcing permutation invariance and consistent behavior.

### 4.2 Centralized Attention Critic

The critic (`CentralAttention1`) implements an **attention-based centralized value function** for exactly three agents:

- Per-agent input: encoded observation, action, and LiDAR embedding.
- Per-agent latent \(h_i\) is passed through multi-head self-attention:
  - Queries, keys, and values are learned projections,
  - Attention weights \(\alpha_{ij}\) capture which agent interactions matter.
- Each agent receives a context vector \(c_i\), and the critic forms \(z_i = [h_i, c_i]\).
- Flattened \(z\) is processed by **twin Q-networks** (`q1`, `q2`) for Soft Actor–Critic.

The critic learns relational structure: which robots are close, which obstacles matter, and how these interactions impact expected return.

### 4.3 SAC Training Loop

The `SAC` class implements the full **Soft Actor–Critic** algorithm:

- **Actor loss**:
  \(J_\pi = \mathbb{E}[\alpha \log\pi_\theta(a \mid o) - Q_\phi(o,a)]\),
- **Critic loss** (soft Bellman backup with target networks),
- **Temperature loss**: adjusting \(\alpha\) toward a target entropy.

Experience is stored in `Ma_Rb_conv` (multi-agent replay buffer) and sampled as flattened (batch × agents) tensors for training the actor and critic.

---

## 5. Curriculum Learning and Stages

Curriculum is implemented via `create_env_for_stage(stage)` in `main.py`, which configures `EnvCoop` as:

1. **Stage 1 — Goal Reaching**
   - 3 robots, **no obstacles**, **no formation constraint**, **no safety filter**.
   - Agents learn basic goal-directed motion and collision avoidance via reward shaping only.

2. **Stage 2 — Formation-Keeping Cooperation**
   - 3 robots, **no obstacles**, **formation enabled**, **no safety filter**.
   - A formation penalty encourages robots to maintain an approximate spacing defined by `dist_robots` while still reaching the goal.

3. **Stage 3 — MPC-Safe Obstacle-Aware Navigation**
   - 3 robots, **obstacles enabled**, **formation enabled**, **MPC safety filter enabled**.
   - The reward now includes obstacle proximity and MPC deviation penalties.
   - The safety layer enforces hard geometric constraints during execution.

A typical workflow is:

1. Train Stage 1 until convergence.
2. Train Stage 2, reusing or fine-tuning from the Stage 1 checkpoint.
3. Train Stage 3 with the MPC filter active.

The exact training schedule can be controlled via `main.py` and saved model prefixes.

---

## 6. Training and Evaluation

### 6.1 Training

Training is orchestrated by `main.py`. In the simplest form:

```bash
python main.py
```

This will:

- Create an `EnvCoop` instance for the selected stage (see `create_env_for_stage`),
- Build the SAC agent and replay buffer,
- Interleave environment interaction and gradient updates,
- Periodically save models into `./models/` (e.g., `sac_epXXXXX_*`),
- Log metrics to TensorBoard under `runs/main_curriculum_3stage/`.

You can customize:

- Number of episodes / total steps,
- Stage selection,
- Model save prefix,
- Random seed,

by editing `main.py` or extending it with argument parsing.

### 6.2 Quantitative Evaluation (`eval_stats.py`)

`eval_stats.py` loads a trained checkpoint and evaluates performance over multiple episodes, optionally with the safety filter enabled:

- Computes **success rate**, **average return**, and **average collisions**,
- Stores results as JSON in `./eval_stats/`, e.g.:
  - `stage1_paper_eval.json`
  - `stage2_paper_eval.json`
  - `stage3_paper_eval.json`

Example usage:

```bash
python eval_stats.py
```

By default, the script:

- Builds a stage-specific environment via `create_env_for_stage(stage)`,
- Loads the SAC agent with matching dimensions,
- Runs a number of evaluation episodes per stage,
- Enforces the safety filter during evaluation when required by the stage.

### 6.3 Visual Debugging (`eval_visual.py`, `eval_video.py`)

- `eval_visual.py` opens an interactive matplotlib window and animates the robots in real time. Useful for qualitative inspection of behaviors (goal-reaching, formation, obstacle avoidance, safety interventions).
- `eval_video.py` records episodes as MP4 files into `./results/`, which can be embedded into the README as demo videos for the three stages.

These scripts rely on the same environment and SAC model but focus on visualization rather than aggregate statistics.

---

## 7. Repository Structure

A suggested structure for this project:

```text
.
├── ATT_modules.py         # Centralized attention critic
├── SAC_ATT.py             # SAC actor + critic + training logic
├── env_coop.py            # Cooperative navigation environment (no Gym)
├── mpc_filter.py          # MPC safety filter (ACADOS + sampling fallback)
├── respawnGoal.py         # Goal management and respawn logic
├── utilis.py              # Utility functions (soft-update, seeding, etc.)
├── main.py                # Training script / curriculum driver
├── eval_stats.py          # Quantitative evaluation (JSON stats)
├── eval_visual.py         # Interactive visual evaluation
├── eval_video.py          # Video recording for demo episodes (optional)
├── models/                # Saved SAC checkpoints
├── eval_stats/            # Stored evaluation JSON files
├── results/               # Saved evaluation videos (MP4)
└── README.md              # This file
```

You can adapt or extend this layout as needed (e.g., add `configs/`, `plots/`, or `paper/` directories).

---

## 8. Dependencies

Core dependencies include:

- Python 3.10+
- `numpy`
- `torch`
- `matplotlib`
- `tensorboard`
- (Optional, for exact MPC) `acados`, `casadi`, and the ACADOS Python interface

If ACADOS is not installed, the repo will **automatically fall back** to the sampling-based safety filter. This behavior is declared at import time in `mpc_filter.py`.

A minimal installation might look like:

```bash
pip install numpy torch matplotlib tensorboard
# Optional: install acados + casadi if you want the exact MPC backend
```

---

## 9. Acknowledgment and Citation

This repository is an **unofficial reimplementation and extension** of the method proposed by Dawood et al. The original work is:

> Murad Dawood, Sicong Pan, Nils Dengler, Siqi Zhou, Angela P. Schoellig, and Maren Bennewitz,
> *Safe Multi-Agent Reinforcement Learning for Behavior-Based Cooperative Navigation*, 2025.

If you build on this repo in academic work, please **cite the original paper** using the authors’ BibTeX:

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

When describing this repository, you may additionally reference it as:

> *Safe MARL Cooperative Navigation (PyTorch Reimplementation)*,
> unofficial reproduction and extension of Dawood et al., 2025.

---

Please treat this code as **research/educational only** and contact the authors of this repo before any commercial use.
