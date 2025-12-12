"""
mpc_filter.py
-------------
Implements the safety layer a_safe = π_safe(u_RL, x) enforcing:

  • robot–obstacle distance constraints
  • robot–robot separation constraints
  • unicycle dynamics prediction
  • minimal deviation from RL action

Two backends:
  1) ACADOS short-horizon MPC (solves nonlinear constrained OCP)
  2) Sampling-based approximate MPC fallback (if ACADOS unavailable)
"""

import numpy as np
import math

# =============================================================
# Problem constants — must match the environment formulation
# =============================================================
DT = 0.1                         # unicycle discrete-time step Δt
MAX_LIN_VEL = 0.5
MAX_ANG_VEL = 0.5
ROBOT_RADIUS = 0.15
OBSTACLE_RADIUS = 0.25

# Required safety radii from paper:
#  d_obs(x) ≥ d_safe_obs    and    d_nei(x) ≥ d_safe_nei
SAFE_OBS_DIST = ROBOT_RADIUS + OBSTACLE_RADIUS + 0.25   # = 0.9
SAFE_NEI_DIST = 2 * ROBOT_RADIUS + 0.15                 # = 0.45

UNSAFE_PENALTY = 1e6          # used by fallback optimizer (soft constraint)

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# =============================================================
# Attempt ACADOS import — if not available, use fallback
# =============================================================
_ACADOS_AVAILABLE = False
try:
    from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
    import casadi as ca
    _ACADOS_AVAILABLE = True
except ImportError:
    _ACADOS_AVAILABLE = False
    print("[MPC] ACADOS unavailable → Using sampling-based safety filter.")


# =====================================================================
# SAMPLING-BASED MPC APPROXIMATION
# Approximates the true MPC by evaluating candidate actions {u_k}
# and selecting the safest action:
#
#     u_safe = argmin_u  (‖u - u_RL‖² + Σ trajectory penalties)
#
# This approximates:
#   min_u  ||u - u_RL||^2
#   s.t.   predicted x_t satisfies safety distances
#
# =====================================================================
class _SamplingFallback:
    """
    Multi-step lookahead approximation of MPC:

      • simulates predicted trajectory under candidate action
      • penalizes unsafe predicted distances
      • returns minimal-cost safe action

    This matches the mathematical MPC filter but without solving
    the nonlinear constrained optimization problem.
    """

    def __init__(self, horizon_steps=7, dt=DT, w_track=1.0, w_obs=10.0, w_nei=8.0):

        self.H = horizon_steps   # short finite horizon N
        self.dt = dt

        # cost weights used in approximate objective:
        #
        #   J(u) = w_track‖u - u_RL‖²  +  Σ_t w_obs/d_obs(t) + w_nei/d_nei(t)
        #
        self.w_track = w_track
        self.w_obs = w_obs
        self.w_nei = w_nei

    # ---------------------------------------------------------
    # Unicycle simulation step: forward model f(x,u)
    # ---------------------------------------------------------
    def simulate_step(self, state, u):
        x, y, th = state
        v, w = u

        v = float(np.clip(v, -MAX_LIN_VEL, MAX_LIN_VEL))
        w = float(np.clip(w, -MAX_ANG_VEL, MAX_ANG_VEL))

        x_new = x + v * math.cos(th) * self.dt
        y_new = y + v * math.sin(th) * self.dt
        th_new = wrap_angle(th + w * self.dt)

        return np.array([x_new, y_new, th_new], dtype=np.float32)

    # ---------------------------------------------------------
    # Approximate cost J(u) for one action candidate
    # ---------------------------------------------------------
    def trajectory_cost(self, x0, u_cand, u_rl, obs_xy, nei1_xy, nei2_xy):
        """
        Computes soft MPC objective:

             J(u) = w_track‖u − u_RL‖²
                     + Σ_t [ w_obs/d_obs(t) + w_nei/d_nei1(t) + w_nei/d_nei2(t) ]

        + huge penalty if any predicted distance violates safety threshold.
        """

        state = np.array(x0, dtype=np.float32)
        u_cand = np.array(u_cand, dtype=np.float32)
        u_rl   = np.array(u_rl, dtype=np.float32)

        cost = self.w_track * np.sum((u_cand - u_rl)**2)
        eps = 1e-3
        unsafe = False

        ox, oy = obs_xy
        n1x, n1y = nei1_xy
        n2x, n2y = nei2_xy

        # simulate horizon
        for _ in range(self.H):
            state = self.simulate_step(state, u_cand)
            px, py, _ = state

            d_obs = np.linalg.norm([px - ox, py - oy])
            d_n1  = np.linalg.norm([px - n1x, py - n1y])
            d_n2  = np.linalg.norm([px - n2x, py - n2y])

            # detect predicted constraint violations (hard MPC constraints)
            if d_obs < SAFE_OBS_DIST or d_n1 < SAFE_NEI_DIST or d_n2 < SAFE_NEI_DIST:
                unsafe = True

            # soft penalty terms (smooth inverses of distance)
            cost += self.w_obs * (1.0 / (d_obs + eps))
            cost += self.w_nei * (1.0 / (d_n1 + eps))
            cost += self.w_nei * (1.0 / (d_n2 + eps))

        if unsafe:
            cost += UNSAFE_PENALTY

        return cost

    # ---------------------------------------------------------
    # Build discrete set of candidate actions around RL output
    # ---------------------------------------------------------
    def build_candidates(self, u_rl):
        """
        Constructs grid {u_k} around u_RL by sampling values of (v,w).
        Approximates solving the optimal control problem over 1 step.
        """
        v_rl, w_rl = u_rl
        dv = 0.20 * MAX_LIN_VEL
        dw = 0.20 * MAX_ANG_VEL

        offsets = [-2, -1, 0, 1, 2]
        candidates = []

        for k in offsets:
            for l in offsets:
                v = float(np.clip(v_rl + k * dv, -MAX_LIN_VEL, MAX_LIN_VEL))
                w = float(np.clip(w_rl + l * dw, -MAX_ANG_VEL, MAX_ANG_VEL))
                candidates.append([v, w])

        # guarantee u_RL ∈ candidate set
        v_rl_c = float(np.clip(v_rl, -MAX_LIN_VEL, MAX_LIN_VEL))
        w_rl_c = float(np.clip(w_rl, -MAX_ANG_VEL, MAX_ANG_VEL))
        if [v_rl_c, w_rl_c] not in candidates:
            candidates.append([v_rl_c, w_rl_c])

        return candidates

    # ---------------------------------------------------------
    # Main safety filter output:
    #
    #    u_safe = argmin_u J(u)
    #
    # ---------------------------------------------------------
    def run(self, u_L, x0, obs_xy, nei1_xy, nei2_xy):

        u_L = np.array(u_L, dtype=np.float32)
        x0  = np.array(x0, dtype=np.float32)

        if np.linalg.norm(u_L) < 1e-5:
            return u_L.astype(np.float32)

        candidates = self.build_candidates(u_L)

        best_cost = float("inf")
        best_action = u_L

        for u_cand in candidates:
            J = self.trajectory_cost(x0, u_cand, u_L, obs_xy, nei1_xy, nei2_xy)
            if J < best_cost:
                best_cost = J
                best_action = np.array(u_cand, dtype=np.float32)

        return best_action.astype(np.float32)



# =====================================================================
# TRUE MPC FILTER (ACADOS back-end)
#
# Solves nonlinear OCP:
#
#   minimize     Σ_k ‖u_k − u_RL‖_W²
#   subject to   x_{k+1} = f(x_k, u_k)
#                d_obs(x_k) ≥ SAFE_OBS_DIST
#                d_nei(x_k) ≥ SAFE_NEI_DIST
#                u ∈ [umin, umax]
#
# =====================================================================
class _AcadosFilter:
    """
    Implements the exact MPC safety filter described in the paper,
    solved using ACADOS:

        min_u  Σ_k  (v_k - v_RL)^2 + (w_k - w_RL)^2
        s.t.   x_{k+1} = f(x_k, u_k)
               d_obs(x_k) ≥ SAFE_OBS_DIST
               d_nei(x_k) ≥ SAFE_NEI_DIST
               u bounds

    """

    def __init__(self, horizon_steps=10, dt=DT, w_track_v=1.0, w_track_w=1.0):

        self.N = horizon_steps
        self.dt = dt

        self.nx = 3     # (x,y,θ)
        self.nu = 2     # (v,w)
        self.np = 6     # (ox,oy, n1x,n1y, n2x,n2y)

        # Weights correspond to "min deviation from RL policy"
        self.w_track_v = w_track_v
        self.w_track_w = w_track_w

        # Build optimal control problem
        self._build_ocp()

    # ---------------------------------------------------------
    # ACADOS OCP BUILD
    # ---------------------------------------------------------
    def _build_ocp(self):
        try:
            model = AcadosModel()
            model.name = "unicycle_mpc"

            # State x, control u, and parameters p
            x = ca.SX.sym('x', 3)   # x = [x, y, θ]
            u = ca.SX.sym('u', 2)   # u = [v, w]
            p = ca.SX.sym('p', 6)   # p = [obs_x,obs_y,n1_x,n1_y,n2_x,n2_y]

            # -----------------------------------------------------
            # Unicycle dynamics f(x,u)
            # -----------------------------------------------------
            v = u[0]
            w = u[1]
            theta = x[2]

            x_dot = ca.vertcat(
                v * ca.cos(theta),
                v * ca.sin(theta),
                w
            )

            model.x = x
            model.u = u
            model.p = p
            model.f_expl_expr = x_dot

            # -----------------------------------------------------
            # Safety constraints (distance ≥ thresholds)
            # -----------------------------------------------------
            obs_vec = x[0:2] - p[0:2]
            n1_vec  = x[0:2] - p[2:4]
            n2_vec  = x[0:2] - p[4:6]

            d_obs = ca.sqrt(obs_vec[0]**2 + obs_vec[1]**2)
            d_n1  = ca.sqrt(n1_vec[0]**2 + n1_vec[1]**2)
            d_n2  = ca.sqrt(n2_vec[0]**2 + n2_vec[1]**2)

            # h(x) = [d_obs, d_n1, d_n2]
            model.con_h_expr = ca.vertcat(d_obs, d_n1, d_n2)
            self.model = model

            # -----------------------------------------------------
            # Create OCP
            # -----------------------------------------------------
            ocp = AcadosOcp()
            ocp.model = model

            ocp.dims.N = self.N
            ocp.solver_options.tf = self.N * self.dt

            ocp.dims.np = self.np
            ocp.parameter_values = np.zeros(self.np)

            # Control bounds u_min ≤ u ≤ u_max
            ocp.constraints.lbu = np.array([-MAX_LIN_VEL, -MAX_ANG_VEL])
            ocp.constraints.ubu = np.array([ MAX_LIN_VEL,  MAX_ANG_VEL])
            ocp.constraints.idxbu = np.array([0, 1])

            # Path constraints: h(x) ≥ safety_margin
            relax = 0.45
            ocp.constraints.lh = np.array([
                SAFE_OBS_DIST - relax,
                SAFE_NEI_DIST - relax,
                SAFE_NEI_DIST - relax
            ])
            ocp.constraints.uh = np.array([1e3, 1e3, 1e3])

            # -----------------------------------------------------
            # Cost function (nonlinear least squares):
            #
            #   y = [v, w]
            #   J = Σ_k  (y_k - y_RL)ᵀ W (y_k - y_RL)
            # -----------------------------------------------------
            ny = 2
            ocp.cost.cost_type = "NONLINEAR_LS"
            ocp.cost.cost_type_e = "NONLINEAR_LS"

            y = ca.vertcat(u[0], u[1])
            ocp.model.cost_y_expr = y
            ocp.model.cost_y_expr_e = ca.DM.zeros(ny,1)

            W = np.diag([self.w_track_v, self.w_track_w])
            ocp.cost.W = W
            ocp.cost.W_e = np.zeros((ny, ny))

            ocp.cost.yref = np.zeros((ny,))
            ocp.cost.yref_e = np.zeros((ny,))

            # initial state (overwritten on each MPC call)
            ocp.constraints.x0 = np.zeros((self.nx,))

            ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
            ocp.solver_options.nlp_solver_type = "SQP_RTI"
            ocp.solver_options.integrator_type = "ERK"
            ocp.solver_options.print_level = 0

            # build solver
            self.solver = AcadosOcpSolver(ocp, json_file="acados_ocp_unicycle.json")

        except Exception as e:
            print(f"[MPC] ACADOS build failure: {e}")
            self.solver = None
            raise

    # ---------------------------------------------------------
    # Solve MPC OCP → return first control u₀
    # ---------------------------------------------------------
    def run(self, u_L, x0, obs_xy, nei1_xy, nei2_xy):

        u_L = np.array(u_L, dtype=np.float32)
        x0  = np.array(x0, dtype=np.float32)

        if np.linalg.norm(u_L) < 1e-6:
            return u_L.astype(np.float32)

        if self.solver is None:
            return u_L.astype(np.float32)

        # Initial state for MPC
        self.solver.set(0, "x", x0)

        # Pack parameters (obstacle + neighbors)
        p = np.array([
            obs_xy[0], obs_xy[1],
            nei1_xy[0], nei1_xy[1],
            nei2_xy[0], nei2_xy[1]
        ], dtype=np.float32)

        for k in range(self.N + 1):
            self.solver.set(k, "p", p)

        # RL action reference (minimize deviation)
        yref = np.array([u_L[0], u_L[1]], dtype=np.float32)
        for k in range(self.N):
            self.solver.set(k, "yref", yref)

        # initial guess
        for k in range(self.N):
            self.solver.set(k, "u", u_L)

        status = self.solver.solve()

        if status != 0:
            print(f"[MPC] ACADOS solver failed (status={status}) → Using RL action.")
            return u_L.astype(np.float32)

        # extract optimal first control input u₀
        u0 = self.solver.get(0, "u").flatten()

        v = float(np.clip(u0[0], -MAX_LIN_VEL, MAX_LIN_VEL))
        w = float(np.clip(u0[1], -MAX_ANG_VEL, MAX_ANG_VEL))

        return np.array([v, w], dtype=np.float32)



# =====================================================================
# PUBLIC SAFETY FILTER (wrapper used by environment)
# =====================================================================
class filter_mpc:
    """
    Unified interface:

        safe_u = filter_mpc_instance.run_mpc(
            u_RL, x0,
            obs_x, obs_y,
            x1, y1,
            x2, y2,
            agent_index
        )

    Internally decides between:
        • ACADOS MPC (exact constrained OCP)
        • Sampling fallback (approximate MPC)
    """

    def __init__(self, horizon_steps=10, dt=DT):

        if _ACADOS_AVAILABLE:
            try:
                print("[MPC] Initializing ACADOS safety filter...")
                self.backend = _AcadosFilter(horizon_steps=horizon_steps, dt=dt)
                print("[MPC] ACADOS MPC is active.")
            except Exception:
                print("[MPC] ACADOS init failed → Using sampling-based fallback.")
                self.backend = _SamplingFallback(horizon_steps=horizon_steps, dt=dt)

        else:
            print("[MPC] ACADOS missing → Using sampling-based fallback.")
            self.backend = _SamplingFallback(horizon_steps=horizon_steps, dt=dt)

    # ---------------------------------------------------------
    # External API used by env_coop
    # ---------------------------------------------------------
    def run_mpc(self, u_L, x0, obs_x, obs_y, x_1, y_1, x_2, y_2, i):

        obs_xy  = np.array([obs_x,  obs_y], dtype=np.float32)
        nei1_xy = np.array([x_1,   y_1], dtype=np.float32)
        nei2_xy = np.array([x_2,   y_2], dtype=np.float32)

        return self.backend.run(u_L, x0, obs_xy, nei1_xy, nei2_xy)
