#!/usr/bin/env python3
"""
eval_visual.py
--------------
Interactive visualization of trained SAC policy.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from env_coop import X_MIN, X_MAX, Y_MIN, Y_MAX, ROBOT_RADIUS, OBSTACLE_RADIUS
from SAC_ATT import SAC
from main import create_env_for_stage


MODEL_PREFIX = "models/sac_ep5000"
EPISODES_PER_STAGE = 10
MAX_STEPS = 600
SCAN_SIZE = 40

MAX_ACTION_V = 0.5
MAX_ACTION_W = 0.5


# ==============================
def load_agent(n_agents):
    agent = SAC(
        seed=0,
        state_dim=n_agents * 12,
        actor_state_dim=12,
        action_dim=2,
        max_action_v=MAX_ACTION_V,
        max_action_w=MAX_ACTION_W,
        scan_size=SCAN_SIZE,
        replay_buffer=None,
        discount=0.99,
        reward_scale=1.0,
        batch_size=1,
    )
    agent.load(MODEL_PREFIX)
    return agent


# ==============================
def draw_env(ax, env):
    ax.clear()
    ax.set_xlim(X_MIN - 0.2, X_MAX + 0.2)
    ax.set_ylim(Y_MIN - 0.2, Y_MAX + 0.2)
    ax.set_aspect("equal")
    ax.set_title("Cooperative Navigation — Visual Eval")

    ax.add_patch(patches.Rectangle(
        (X_MIN, Y_MIN),
        X_MAX - X_MIN,
        Y_MAX - Y_MIN,
        fill=False, edgecolor="black"
    ))

    for (ox, oy, _) in env.obstacles:
        ax.add_patch(plt.Circle((ox, oy), OBSTACLE_RADIUS, color="grey", alpha=0.5))

    gx, gy = env.goal
    ax.add_patch(plt.Circle((gx, gy), 0.25, color="green", alpha=0.4))

    for i, pos in enumerate(env.position):
        ax.add_patch(plt.Circle(pos, ROBOT_RADIUS, color="blue"))
        ax.text(pos[0], pos[1], str(i), color="white",
                ha="center", va="center", fontsize=8)


# ==============================
def run_visual(stage):
    env = create_env_for_stage(stage)
    agent = load_agent(env.n_agents)

    print(f"\n=== VISUALIZING STAGE {stage} ===\n")

    for ep in range(1, EPISODES_PER_STAGE + 1):
        _, astate, scan = env.reset()

        fig, ax = plt.subplots(figsize=(6, 6))
        plt.ion()
        plt.show()

        print(f"Episode {ep}")

        for step in range(MAX_STEPS):
            draw_env(ax, env)
            plt.pause(0.001)

            action = agent.select_action(astate, scan, eval=True)
            _, astate, _, dones, scan, _ = env.step(action)

            if dones[0]:
                print(f"  Ended | Goals={env.goals_cnt} | Collisions={env.cols}")
                break

        plt.ioff()
        plt.close(fig)


# ==============================
def main():
    for stage in [1,2,3]:
        run_visual(stage)

    print("\n=== Visual Evaluation Complete ===\n")


if __name__ == "__main__":
    main()
