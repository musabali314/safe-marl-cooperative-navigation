#!/usr/bin/env python3
"""
eval_stats.py
-------------
Paper-grade evaluation for SAC + Attention + MPC in EnvCoop.

Reports:
• Success rate
• Timeout rate
• Collision rate
• Avg steps-to-goal
• Avg return

MPC is applied ONLY inside EnvCoop.
"""

import json
import os
import numpy as np

from SAC_ATT import SAC
from main import create_env_for_stage


# ================================================================
# CONFIG (MATCH ENV_COOP)
# ================================================================
MODEL_PREFIX = "models/sac_ep5000"
EPISODES_PER_STAGE = 50
MAX_STEPS = 600          # allow safe policies to finish
SCAN_SIZE = 40

MAX_ACTION_V = 0.5
MAX_ACTION_W = 0.5


# ================================================================
def load_agent_for_stage(n_agents):

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
    print("[EVAL] Loaded model:", MODEL_PREFIX)
    return agent


# ================================================================
def evaluate_stage(stage, episodes):

    env = create_env_for_stage(stage)
    agent = load_agent_for_stage(env.n_agents)

    success = 0
    collisions = 0
    timeouts = 0

    all_returns = []
    steps_to_goal = []

    print(f"\n=== Evaluating Stage {stage} ===")

    for ep in range(episodes):

        _, actor_state, scan = env.reset()
        ep_return = 0.0
        reached_goal = False

        for step in range(MAX_STEPS):

            action = agent.select_action(actor_state, scan, eval=True)

            (
                _,
                next_actor_state,
                rewards,
                dones,
                next_scan,
                reward_info
            ) = env.step(action)

            ep_return += float(rewards.mean())
            actor_state = next_actor_state
            scan = next_scan

            if reward_info["goals"] > 0:
                reached_goal = True
                success += 1
                steps_to_goal.append(step + 1)
                break

            if reward_info["collisions"] > 0:
                collisions += 1
                break

            if dones[0]:
                break

        if not reached_goal and env.cols == 0:
            timeouts += 1

        all_returns.append(ep_return)

    # ============================================================
    results = {
        "stage": stage,
        "episodes": episodes,
        "success_rate": success / episodes,
        "collision_rate": collisions / episodes,
        "timeout_rate": timeouts / episodes,
        "avg_steps_to_goal": float(np.mean(steps_to_goal)) if steps_to_goal else None,
        "avg_return": float(np.mean(all_returns)),
    }

    return results


# ================================================================
def main():

    os.makedirs("eval_stats", exist_ok=True)

    for stage in [3]:
        results = evaluate_stage(stage, EPISODES_PER_STAGE)

        fname = f"eval_stats/stage{stage}_paper_eval.json"
        with open(fname, "w") as f:
            json.dump(results, f, indent=4)

        print(f"→ Saved {fname}")

    print("\n=== Paper-Grade Evaluation Complete ===")


if __name__ == "__main__":
    main()
