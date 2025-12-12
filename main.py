#!/usr/bin/env python
import os
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from env_coop import EnvCoop, LIDAR_RAYS
from utilis import Ma_Rb_conv
from SAC_ATT import SAC, ACTION_V_MAX, ACTION_W_MAX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=== DEVICE INFO ===")
print("Using:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
print("===================")


# ============================================================
# Hyperparameters
# ============================================================
SEED = 0
NUM_EPISODES = 100000
MAX_EPISODE_STEPS = 750

START_TRAIN_AFTER = 5000
TRAIN_EVERY = 1
EVAL_INTERVAL = 300
SAVE_INTERVAL = 5000

DISCOUNT = 0.99
REWARD_SCALE = 1.0

REPLAY_CAPACITY = int(1e6)
BATCH_SIZE = 256

MAX_STAGE = 3


# ============================================================
def create_env_for_stage(stage: int) -> EnvCoop:
    if stage == 1:
        return EnvCoop(12, 3, False, False, False)
    elif stage == 2:
        return EnvCoop(12, 3, False, True, False)
    elif stage == 3:
        return EnvCoop(12, 3, True, True, True)
    else:
        raise ValueError(f"Invalid stage: {stage}")


# ============================================================
def create_agent_and_buffer(scan_size):
    critic_dim = 3 * 12
    buffer = Ma_Rb_conv(
        critic_dim=critic_dim,
        actor_dim=12,
        action_dim=2,
        num_agents=3,
        scan_size=scan_size,
        max_size=REPLAY_CAPACITY,
    )

    agent = SAC(
        seed=SEED,
        state_dim=critic_dim,
        actor_state_dim=12,
        action_dim=2,
        max_action_v=ACTION_V_MAX,
        max_action_w=ACTION_W_MAX,
        scan_size=scan_size,
        replay_buffer=buffer,
        discount=DISCOUNT,
        reward_scale=REWARD_SCALE,
        batch_size=BATCH_SIZE,
    )
    return agent, buffer


# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
def evaluate_policy(env, agent, episodes=10):
    success = 0
    returns = []
    collisions = []

    for _ in range(episodes):
        env.goals_cnt = 0  # reset goal counter

        cstate, astate, scan = env.reset()
        ep_ret = 0.0

        for _ in range(MAX_EPISODE_STEPS):
            act = agent.select_action(astate, scan, eval=True)
            cnext, anext, rew, done, scnext, _ = env.step(act)

            ep_ret += float(rew.mean())
            astate, scan = anext, scnext

            if done[0]:
                break

        returns.append(ep_ret)
        collisions.append(env.cols)
        success += int(env.goals_cnt >= 1)

    return {
        "success_rate": success / episodes,
        "avg_return": np.mean(returns),
        "avg_collisions": np.mean(collisions),
    }


# ============================================================
# Main Training Loop
# ============================================================
def main():
    set_seed(SEED)
    writer = SummaryWriter("runs/main_curriculum_3stage/")

    stage = 3
    env = create_env_for_stage(stage)

    agent, replay_buffer = create_agent_and_buffer(LIDAR_RAYS)
    os.makedirs("models", exist_ok=True)

    total_steps = 0
    last_eval_sr = 0.0

    eval_history = {
        "episode": [], "stage": [],
        "success_rate": [], "avg_return": [], "avg_collisions": []
    }

    stage_min_ep = {1: 0, 2: 0, 3: 0}

    # ============================================================
    # For tracking reward-component sums every episode
    # ============================================================
    comp_names = ["alive", "prog", "goal_shape", "form", "rr", "obs", "mpc",
                  "collisions", "goals"]

    for ep in range(1, NUM_EPISODES + 1):

        # ---------------- Evaluation & Stage Advancement ----------------
        if ep % EVAL_INTERVAL == 0:
            metrics = evaluate_policy(env, agent, episodes=5)
            last_eval_sr = metrics["success_rate"]

            print(f"[EVAL] Ep {ep} | Stage {stage} | "
                  f"SR={last_eval_sr:.2f} | Ret={metrics['avg_return']:.2f} "
                  f"| Coll={metrics['avg_collisions']:.2f}")

            writer.add_scalar("Eval/SuccessRate", metrics["success_rate"], ep)
            writer.add_scalar("Eval/AvgReturn", metrics["avg_return"], ep)
            writer.add_scalar("Eval/AvgCollisions", metrics["avg_collisions"], ep)

            eval_history["episode"].append(ep)
            eval_history["stage"].append(stage)
            eval_history["success_rate"].append(last_eval_sr)
            eval_history["avg_return"].append(metrics["avg_return"])
            eval_history["avg_collisions"].append(metrics["avg_collisions"])

            if stage < MAX_STAGE and ep >= stage_min_ep[stage + 1] and last_eval_sr > 0.7:
                new_stage = stage + 1
                print(f"\n====== ADVANCING {stage} → {new_stage} ======\n")
                stage = new_stage
                env = create_env_for_stage(stage)

                critic_dim = 3 * 12
                replay_buffer = Ma_Rb_conv(
                    critic_dim=critic_dim,
                    actor_dim=12,
                    action_dim=2,
                    num_agents=3,
                    scan_size=LIDAR_RAYS,
                    max_size=REPLAY_CAPACITY,
                )
                agent.replay_buffer = replay_buffer

        # ---------------- Episode Training ----------------
        critic_state, actor_states, scans = env.reset()
        ep_rew = 0.0
        ep_steps = 0

        # reward component accumulators
        ep_reward_info = {k: 0.0 for k in comp_names}

        while True:
            ep_steps += 1
            total_steps += 1

            # random warmup
            if total_steps < START_TRAIN_AFTER:
                rand = np.random.uniform(-1.0, 1.0, size=(3, 2))
                action = np.zeros_like(rand)
                action[:, 0] = rand[:, 0] * ACTION_V_MAX
                action[:, 1] = rand[:, 1] * ACTION_W_MAX
            else:
                action = agent.select_action(actor_states, scans, eval=False)

            critic_next, actor_next, reward, done, scan_next, rinfo = env.step(action)
            ep_rew += float(reward.mean())

            # accumulate reward terms
            for k in comp_names:
                ep_reward_info[k] += rinfo[k]

            replay_buffer.add(
                state_critic=critic_state[0],
                state_actor=actor_states,
                scan=scans,
                next_scan=scan_next,
                action=action,
                next_state_critic=critic_next[0],
                next_state_actor=actor_next,
                reward=reward,
                done=done.astype(np.float32),
            )

            critic_state = critic_next
            actor_states = actor_next
            scans = scan_next

            if replay_buffer.len() >= START_TRAIN_AFTER and total_steps % TRAIN_EVERY == 0:
                actor_l, critic_l, alpha_val = agent.train()
                writer.add_scalar("Loss/Actor", actor_l, total_steps)
                writer.add_scalar("Loss/Critic", critic_l, total_steps)
                writer.add_scalar("Loss/Alpha", alpha_val, total_steps)

            if done[0] or ep_steps >= MAX_EPISODE_STEPS:
                break

        # ---------------- Logging Episode Summary ----------------
        print(f"\nEpisode {ep}/{NUM_EPISODES} | Stage {stage}")
        print(f"  Return={ep_rew:.2f} | Goals={env.goals_cnt} | Collisions={env.cols} | Steps={ep_steps}")

        # ---- Reward Breakdown Pretty Print ----
        print("  Reward Breakdown →")
        print(f"      Alive:          {ep_reward_info['alive']:.3f}")
        print(f"      Progress:       {ep_reward_info['prog']:.3f}")
        print(f"      Goal shaping:   {ep_reward_info['goal_shape']:.3f}")
        print(f"      Formation:      {ep_reward_info['form']:.3f}")
        print(f"      Inter-robot:    {ep_reward_info['rr']:.3f}")
        print(f"      Obstacle:       {ep_reward_info['obs']:.3f}")
        print(f"      MPC dev.:       {ep_reward_info['mpc']:.3f}")
        print(f"      Goal events:    {int(ep_reward_info['goals'])}")
        print(f"      Collision evt.: {int(ep_reward_info['collisions'])}")
        print("------------------------------------------------------------\n")


        # main episode logs
        writer.add_scalar("EpisodeReward", ep_rew, ep)
        writer.add_scalar("EpisodeLength", ep_steps, ep)
        writer.add_scalar("Collisions/Episode", env.cols, ep)
        writer.add_scalar("Goals/Episode", env.goals_cnt, ep)

        # reward-component logs
        for k in comp_names:
            writer.add_scalar(f"RewardComponents/{k}", ep_reward_info[k], ep)

        writer.add_scalar("Debug/ActionStd", float(np.std(action)), ep)

        if ep % SAVE_INTERVAL == 0:
            agent.save(f"models/sac_ep{ep}")
            print(f"Saved model at models/sac_ep{ep}")

    agent.save("models/sac_final")
    np.save("eval_history.npy", eval_history)
    print("Training complete.")


if __name__ == "__main__":
    main()
