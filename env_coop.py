import numpy as np
import math
import random

from respawnGoal import RespawnGoal
from mpc_filter import filter_mpc


# =====================================================================
# GLOBAL CONSTANTS — match mathematical formulation
# =====================================================================
DT = 0.1                     # Δt in the unicycle model
MAX_STEPS = 350              # episode horizon

# Workspace bounds Ω ⊂ ℝ²
X_MIN, X_MAX = -3.0, 3.0
Y_MIN, Y_MAX = -3.0, 3.0

# Robot and obstacle geometry for collision model
ROBOT_RADIUS = 0.15
OBSTACLE_RADIUS = 0.25
N_OBSTACLES = 3

# LiDAR forward model parameters
LIDAR_RANGE = 2.5
LIDAR_RAYS = 40

# Action limits for (v, ω)
MAX_LIN_VEL = 0.5
MAX_ANG_VEL = 0.5


# =====================================================================
# Utility functions: angle wrapping, Euclidean distance, random sampling
# =====================================================================
def wrap_angle(a):
    """
    Implements angle normalization into (-π, π].
    This corresponds to the constraint θ ∈ S¹.
    """
    return (a + np.pi) % (2 * np.pi) - np.pi

def distance(a, b):
    """Euclidean distance ‖a - b‖₂."""
    return np.linalg.norm(a - b)

def sample_xy(margin=0.4):
    """
    Samples a feasible position in Ω with boundary margin.
    Used for robot placement and obstacle placement.
    """
    return np.array([
        random.uniform(X_MIN + margin, X_MAX - margin),
        random.uniform(Y_MIN + margin, Y_MAX - margin)
    ], dtype=np.float32)



# =====================================================================
# MAIN ENVIRONMENT — Implements:
#
#   • Unicycle dynamics
#   • Local observation φ_i(o_i^t)
#   • Reward function r^t = Σ components
#   • LiDAR forward model
#   • Safety filter a_safe = MPC(a_RL)
#   • Curriculum (formation, obstacles, safety filter)
#
# =====================================================================
class EnvCoop:
    def __init__(
        self,
        actor_state_dim=12,
        num_agents=1,
        enable_obstacles=False,
        enable_formation=False,
        enable_safety_filter=False
    ):
        """
        Initializes:
            • N-agent system
            • Feature memory
            • Optional formation constraint
            • Optional obstacles
            • Optional MPC safety filter
        """

        self.actor_state_dim = actor_state_dim
        self.n_agents = num_agents

        self.enable_obstacles = enable_obstacles
        self.enable_formation = enable_formation
        self.enable_safety_filter = enable_safety_filter

        self._allocate_arrays()

        print(
            f"[ENV] Agents={self.n_agents}, Formation={self.enable_formation}, "
            f"Obstacles={self.enable_obstacles}, Safety={self.enable_safety_filter}"
        )

        self.reset()


    # =================================================================
    # MEMORY ALLOCATION
    # =================================================================
    def _allocate_arrays(self):
        """
        Allocate:
            • Pose representation x_i = (x, y, θ)
            • LiDAR scan arrays
            • Previous actions for encoding
            • Stuck counters (for termination)
            • Formation spacing reference
        """

        # Robot states
        self.poses = np.zeros((self.n_agents, 3), dtype=np.float32)
        self.position = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.yaw = np.zeros(self.n_agents, dtype=np.float32)

        # LiDAR
        self.scan = np.zeros((self.n_agents, LIDAR_RAYS), dtype=np.float32)

        # Closest obstacle (needed for MPC filter input)
        self.obst_dist = np.zeros(self.n_agents * 2, dtype=np.float32)

        # Memory for stuck detection + past action
        self.past_position = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.same_pos_counter = np.zeros(self.n_agents, dtype=np.int32)
        self.past_action = np.zeros((self.n_agents, 2), dtype=np.float32)

        # Formation spacing reference d_form ∈ {1.0,1.25,1.5}
        self.dist_options = [1.0, 1.25, 1.5]
        self.dist_robots = 1.0

        # Obstacles + goal
        self.obstacles = []
        self.goal_manager = RespawnGoal()
        self.goal = np.zeros(2, dtype=np.float32)

        # Optional MPC filters (one per agent)
        if self.enable_safety_filter:
            self.filter_mpc = [filter_mpc() for _ in range(self.n_agents)]
        else:
            self.filter_mpc = [None] * self.n_agents

        # Episode counters
        self.step_count = 0
        self.goals_cnt = 0
        self.cols = 0
        self.goal_distance = 0.0
        self.prev_goal_distance = 0.0


    # =================================================================
    # RESET EPISODE
    # =================================================================
    def reset(self):
        """
        Resets entire environment.
        Returns padded (3-agent) critic/actor states and LiDAR scans.
        """

        self.step_count = 0
        self.goals_cnt = 0
        self.cols = 0

        # Randomize formation target spacing
        self.dist_robots = random.choice(self.dist_options)

        # Place robots with no initial collision
        self._sample_robot_poses()

        # Reset memories
        self.same_pos_counter[:] = 0
        self.past_position[:] = self.position
        self.past_action[:] = 0.0

        # Obstacles
        if self.enable_obstacles:
            self._sample_obstacles()
        else:
            self.obstacles = []

        # New goal
        gx, gy = self.goal_manager.getPosition(
            robot_positions=self.position,
            obstacles=self.obstacles,
            delete=True
        )
        self.goal[:] = [gx, gy]

        # Update LiDAR, centroid dist
        self._update_global_state_and_scans()
        self.prev_goal_distance = self.goal_distance

        # Build actor & critic inputs
        actor_states = np.stack([self._get_actor_state(i)[0]
                                for i in range(self.n_agents)])
        scans = self.scan / LIDAR_RANGE

        # Pad to 3 agents for critic
        actor_states = self._pad_agents(actor_states)
        scans = self._pad_agents(scans)

        critic_state = actor_states.reshape(1, -1)

        return (
            critic_state.astype(np.float32),
            actor_states.astype(np.float32),
            scans.astype(np.float32)
        )


    # =================================================================
    # STEP — Implements:
    #
    #   a_safe = MPC(a_RL)
    #
    #   x_{t+1} = f(x_t, a_safe)
    #
    #   r_t = Σ reward components
    #
    # =================================================================
    def step(self, rl_actions):

        self.step_count += 1
        prev_goal_distance = self.goal_distance

        # ------------------------------------------------------------
        # SAFETY FILTER — applies MPC correction if enabled
        # ------------------------------------------------------------
        safe_actions = np.zeros_like(rl_actions)
        for i in range(self.n_agents):

            if self.enable_safety_filter:
                # nearest neighbors and obstacle geometry supplied to MPC
                n1, n2 = self._get_two_neighbors(i)
                ox, oy = self.obst_dist[2*i], self.obst_dist[2*i + 1]

                safe_actions[i] = self.filter_mpc[i].run_mpc(
                    rl_actions[i],
                    self.poses[i].copy(),
                    ox, oy,
                    n1[0], n1[1],
                    n2[0], n2[1],
                    i
                )
            else:
                safe_actions[i] = rl_actions[i]


        # ------------------------------------------------------------
        # APPLY UNICYCLE DYNAMICS
        #
        #   x_{t+1} = x_t + v cosθ Δt
        #   y_{t+1} = y_t + v sinθ Δt
        #   θ_{t+1} = θ_t + ω Δt
        # ------------------------------------------------------------
        for i in range(self.n_agents):
            v, w = safe_actions[i]
            x, y, th = self.poses[i]

            self.poses[i] = [
                x + v * math.cos(th) * DT,
                y + v * math.sin(th) * DT,
                wrap_angle(th + w * DT)
            ]

        # store previous action for next observation
        self.past_action[:] = safe_actions

        # recompute LiDAR + centroid distance + stuck detection
        self._update_global_state_and_scans()

        # per-agent reward computation
        actor_states = []
        rewards = []
        dones_list = []

        # breakdown dictionary for debugging
        reward_info = {
            "alive": 0.0, "prog": 0.0, "goal_shape": 0.0, "form": 0.0,
            "rr": 0.0, "obs": 0.0, "mpc": 0.0,
            "collisions": 0, "goals": 0
        }

        for i in range(self.n_agents):

            s_i, done_i = self._get_actor_state(i)

            (r_i, done_i,
             r_alive, r_prog, r_goal, r_form, r_rr, r_obs, r_mpc,
             is_collision, is_goal) = self._compute_reward(
                 i, done_i, rl_actions, safe_actions, prev_goal_distance
            )

            actor_states.append(s_i)
            rewards.append(r_i)
            dones_list.append(done_i)

            # accumulate component-level reward logs
            reward_info["alive"] += r_alive
            reward_info["prog"] += r_prog
            reward_info["goal_shape"] += r_goal
            reward_info["form"] += r_form
            reward_info["rr"] += r_rr
            reward_info["obs"] += r_obs
            reward_info["mpc"] += r_mpc
            reward_info["collisions"] += is_collision
            reward_info["goals"] += is_goal

        # update reference for next step
        self.prev_goal_distance = self.goal_distance

        actor_states = np.stack(actor_states)
        scans = self.scan / LIDAR_RANGE
        rewards = np.array(rewards, dtype=np.float32)

        # pad everything to 3 agents for critic input
        actor_states = self._pad_agents(actor_states)
        scans = self._pad_agents(scans)
        rewards = self._pad_agents(rewards, pad_last_dim=False)

        critic_state = actor_states.reshape(1, -1)
        done_all = np.any(dones_list) or (self.step_count >= MAX_STEPS)
        dones = np.array([done_all] * 3)

        return (
            critic_state.astype(np.float32),
            actor_states.astype(np.float32),
            rewards,
            dones.astype(np.bool_),
            scans.astype(np.float32),
            reward_info
        )


    # =================================================================
    # Zero-padding helper for critic (always expects 3 agents)
    # =================================================================
    def _pad_agents(self, arr, pad_last_dim=True):
        """
        Pads arrays of shape (1,D) or (2,D) to (3,D) with zeros.
        """
        n = arr.shape[0]
        if n == 3:
            return arr

        if arr.ndim == 1:
            return np.pad(arr, ((0, 3-n),), mode='constant')

        if pad_last_dim:
            D = arr.shape[1]
            padded = np.zeros((3, D), dtype=np.float32)
        else:
            padded = np.zeros((3,), dtype=np.float32)

        padded[:n] = arr
        return padded


    # =================================================================
    # Sampling robot poses (must ensure no initial overlaps)
    # =================================================================
    def _sample_robot_poses(self):
        self.poses[:] = 0.0

        for i in range(self.n_agents):
            while True:
                xy = sample_xy()
                if all(
                    distance(xy, self.poses[j, :2]) > (2 * ROBOT_RADIUS + 0.5)
                    for j in range(i)
                ):
                    th = random.uniform(-np.pi, np.pi)
                    self.poses[i] = [xy[0], xy[1], th]
                    break

        self.position[:] = self.poses[:, :2]
        self.past_position[:] = self.position


    # =================================================================
    # Sampling obstacles with minimum clearance
    # =================================================================
    def _sample_obstacles(self):
        self.obstacles = []
        for _ in range(N_OBSTACLES):
            while True:
                xy = sample_xy()
                if all(
                    distance(xy, pos) > (OBSTACLE_RADIUS + 0.8)
                    for pos in self.position
                ):
                    self.obstacles.append((xy[0], xy[1], OBSTACLE_RADIUS))
                    break


    # =================================================================
    # GLOBAL STATE UPDATE:
    #
    #  • Compute centroid → d_goal(t)
    #  • Compute LiDAR scan
    #  • Track stuck robots
    #
    # =================================================================
    def _update_global_state_and_scans(self):
        self.position[:] = self.poses[:, :2]

        # centroid distance to goal (used for shaping + progress reward)
        centroid = np.mean(self.position, axis=0)
        self.goal_distance = distance(centroid, self.goal)

        # per-agent goal heading (optional logging)
        for i in range(self.n_agents):
            dx, dy = self.goal - self.position[i]
            ang = math.atan2(dy, dx)
            self.yaw[i] = self.poses[i, 2]
            self.heading_rob_goal = wrap_angle(ang - self.yaw[i])

        # LiDAR simulation + nearest obstacle coordinates
        for i in range(self.n_agents):
            scan_i, closest = self._compute_lidar(i)
            self.scan[i] = scan_i
            self.obst_dist[2*i:2*i+2] = closest

        # stuck detection: robot hasn't moved significantly
        for i in range(self.n_agents):
            if np.allclose(self.position[i], self.past_position[i], atol=0.01):
                self.same_pos_counter[i] += 1
            else:
                self.same_pos_counter[i] = 0
            self.past_position[i] = self.position[i].copy()


    # =================================================================
    # LIDAR FORWARD MODEL — simulates ray casting
    # =================================================================
    def _compute_lidar(self, idx):
        """
        Computes:
            • ranges[k] = length of ray until obstacle/wall
            • closest_world = nearest obstacle hit

        This corresponds to computing for each beam k:

            r_k = min t ≥ 0
                  s.t.  x(t) ∉ Ω  or  x(t) ∈ obstacle

        """

        x, y, th = self.poses[idx]
        min_dist = float("inf")
        closest_world = np.array([x, y], dtype=np.float32)

        ranges = np.ones(LIDAR_RAYS, dtype=np.float32) * LIDAR_RANGE

        for k in range(LIDAR_RAYS):
            ang = -np.pi + k * (2 * np.pi / LIDAR_RAYS)
            dir_vec = np.array([
                math.cos(th + ang),
                math.sin(th + ang)
            ], dtype=np.float32)

            hit_t = LIDAR_RANGE
            for t in np.linspace(0.0, LIDAR_RANGE, 50):

                px = x + dir_vec[0] * t
                py = y + dir_vec[1] * t

                # boundary check
                if px < X_MIN or px > X_MAX or py < Y_MIN or py > Y_MAX:
                    hit_t = t
                    break

                # obstacles
                for (ox, oy, r) in self.obstacles:
                    if distance(np.array([px, py]), np.array([ox, oy])) <= r:
                        hit_t = t
                        break

                if hit_t != LIDAR_RANGE:
                    break

            ranges[k] = hit_t

            if hit_t < min_dist:
                min_dist = hit_t
                closest_world = np.array([
                    x + dir_vec[0] * min_dist,
                    y + dir_vec[1] * min_dist
                ], dtype=np.float32)

        return ranges, closest_world


    # =================================================================
    # Neighbor function (needed for MPC’s pairwise constraints)
    # =================================================================
    def _get_two_neighbors(self, idx):
        """
        Returns the two nearest neighbors' poses.
        If not enough agents, duplicates are returned.
        """

        if self.n_agents == 1:
            return self.poses[idx], self.poses[idx]

        elif self.n_agents == 2:
            j = 1 - idx
            return self.poses[j], self.poses[j]

        else:
            others = [i for i in range(self.n_agents) if i != idx]
            return self.poses[others[0]], self.poses[others[1]]


    # =================================================================
    # OBSERVATION FUNCTION — constructs 12D per-agent vector φ(o_i^t)
    # =================================================================
    def _get_actor_state(self, idx):
        """
        Constructs the normalized 12D observation vector:

          o_i = [
            min_scan, ang_closest,
            dist1, ang1, dist2, ang2,
            dist_goal_i, heading_goal_i,
            v_prev, w_prev,
            centroid_goal_dist, formation_ref
          ]
        """

        x_i, y_i, th_i = self.poses[idx]
        pos_i = self.position[idx]

        # --------------------------------------------------------
        # LiDAR features
        # --------------------------------------------------------
        ranges = self.scan[idx]
        min_scan = float(np.min(ranges)) / LIDAR_RANGE

        k = int(np.argmin(ranges))
        ang_closest = -np.pi + k * (2 * np.pi / LIDAR_RAYS)
        ang_closest = wrap_angle(ang_closest - th_i) / np.pi

        # --------------------------------------------------------
        # Robot–robot geometric features
        # --------------------------------------------------------
        if self.n_agents == 1:
            dist1 = dist2 = 1.0
            ang1 = ang2 = 0.0

        else:
            diffs = self.position - pos_i
            dists = np.linalg.norm(diffs, axis=1)
            order = np.argsort(dists)
            nbrs = [o for o in order if o != idx][:2]

            dv1 = self.position[nbrs[0]] - pos_i
            dist1 = float(np.linalg.norm(dv1)) / 3.0
            ang1 = wrap_angle(math.atan2(dv1[1], dv1[0]) - th_i) / np.pi

            if len(nbrs) >= 2:
                dv2 = self.position[nbrs[1]] - pos_i
                dist2 = float(np.linalg.norm(dv2)) / 3.0
                ang2 = wrap_angle(math.atan2(dv2[1], dv2[0]) - th_i) / np.pi
            else:
                dist2 = 1.0
                ang2 = 0.0

        # --------------------------------------------------------
        # Goal features
        # --------------------------------------------------------
        dg = self.goal - pos_i
        dist_goal_agent = float(np.linalg.norm(dg)) / 3.0
        heading_to_goal = wrap_angle(math.atan2(dg[1], dg[0]) - th_i) / np.pi

        # --------------------------------------------------------
        # Previous action
        # --------------------------------------------------------
        v_prev = self.past_action[idx, 0] / MAX_LIN_VEL
        w_prev = self.past_action[idx, 1] / MAX_ANG_VEL

        # --------------------------------------------------------
        # Additional features: centroid-goal & formation ref
        # --------------------------------------------------------
        goal_dist_norm = self.goal_distance / 3.0
        rref_norm = self.dist_robots / 1.5

        # --------------------------------------------------------
        # Terminal flag (collision or stuck)
        # --------------------------------------------------------
        done = self._check_collision(idx) or (self.same_pos_counter[idx] > 30)

        state = np.array([
            min_scan,
            ang_closest,
            dist1,
            ang1,
            dist2,
            ang2,
            dist_goal_agent,
            heading_to_goal,
            v_prev,
            w_prev,
            goal_dist_norm,
            rref_norm,
        ], dtype=np.float32)

        return state, done


    # =================================================================
    # REWARD FUNCTION — Implements the full shaped reward:
    #
    #   r = r_alive + r_prog + r_goal + r_form
    #       + r_rr + r_obs + r_mpc
    #
    # =================================================================
    def _compute_reward(self, idx, done, rl_actions, safe_actions, prev_goal_distance):

        # ------------------------------------------------------------
        # Alive reward
        # ------------------------------------------------------------
        r_alive = 0.04

        # ------------------------------------------------------------
        # Progress reward
        # Δd = d_prev - d_current
        # ------------------------------------------------------------
        delta = prev_goal_distance - self.goal_distance
        r_prog = 10.0 * delta

        # ------------------------------------------------------------
        # Goal shaping negative reward
        # ------------------------------------------------------------
        r_goal = -0.01 * self.goal_distance

        # ------------------------------------------------------------
        # Formation penalty
        # ------------------------------------------------------------
        r_form = 0.0
        if self.enable_formation and self.n_agents == 3:

            neigh = {0:[1,2], 1:[0,2], 2:[0,1]}[idx]

            d1 = np.linalg.norm(self.position[idx] - self.position[neigh[0]])
            d2 = np.linalg.norm(self.position[idx] - self.position[neigh[1]])

            err = abs(d1 - self.dist_robots) + abs(d2 - self.dist_robots)
            r_form = -0.005 * err

        # ------------------------------------------------------------
        # Robot–robot proximity penalty
        # ------------------------------------------------------------
        r_rr = 0.0
        SOFT_DIST = 0.5
        HARD_DIST = 0.3

        min_dist = min([
            np.linalg.norm(self.position[idx] - self.position[j])
            for j in range(self.n_agents) if j != idx
        ] + [999])

        if min_dist < SOFT_DIST:

            if min_dist > HARD_DIST:
                # smooth cosine penalty
                x = (min_dist - SOFT_DIST) / (HARD_DIST - SOFT_DIST)
                r_rr = -0.3 * (1 + np.cos(np.pi * x))
            else:
                # harsh penalty inside hard radius
                r_rr = -1.0 - 2.0 * (HARD_DIST - min_dist)

        # ------------------------------------------------------------
        # Obstacle proximity penalty
        # ------------------------------------------------------------
        if self.enable_obstacles:
            min_scan = float(np.min(self.scan[idx]))
            SAFE_OBS_BAND = 1.0
            dist_to_safe = max(0.0, SAFE_OBS_BAND - min_scan)
            r_obs = -0.25 * (dist_to_safe / SAFE_OBS_BAND)
        else:
            r_obs = 0.0

        # ------------------------------------------------------------
        # MPC deviation penalty
        # ------------------------------------------------------------
        dv = abs(rl_actions[idx, 0] - safe_actions[idx, 0])
        dw = abs(rl_actions[idx, 1] - safe_actions[idx, 1])

        r_mpc = -0.6 * (dv + dw) if self.enable_safety_filter else 0.0

        # ------------------------------------------------------------
        # Combine shaping rewards
        # ------------------------------------------------------------
        reward = (
            r_alive +
            r_prog +
            r_goal +
            r_form +
            r_rr +
            r_obs +
            r_mpc
        )

        # ------------------------------------------------------------
        # TERMINAL EVENTS (goal or collision)
        # ------------------------------------------------------------
        centroid = np.mean(self.position, axis=0)

        is_goal = (np.linalg.norm(centroid - self.goal) < 0.25)
        is_collision = (self._check_collision(idx) or self.same_pos_counter[idx] > 30)

        if is_goal:
            reward = 50.0
            done = True
            self.goals_cnt += 1

            gx, gy = self.goal_manager.getPosition(
                self.position, self.obstacles, delete=True
            )
            self.goal[:] = [gx, gy]

        elif is_collision:
            reward = -100.0
            done = True
            self.cols += 1

        return (
            float(reward), done,
            r_alive, r_prog, r_goal, r_form, r_rr, r_obs, r_mpc,
            int(is_collision), int(is_goal)
        )


    # =================================================================
    # Collision checking function
    # =================================================================
    def _check_collision(self, idx):

        pos = self.position[idx]

        # robot–obstacle
        for (ox, oy, r) in self.obstacles:
            if distance(pos, np.array([ox, oy], dtype=np.float32)) < (r + ROBOT_RADIUS):
                return True

        # robot–robot
        for j in range(self.n_agents):
            if j == idx:
                continue
            if distance(pos, self.position[j]) < 2 * ROBOT_RADIUS:
                return True

        return False
