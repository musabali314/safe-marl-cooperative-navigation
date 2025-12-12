import numpy as np
import torch
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# Multi-Agent Replay Buffer
# ==========================================================
class Ma_Rb_conv(object):
    """
    Multi-agent replay buffer.
    Stores:
        • critic_state      (B, critic_dim)
        • actor_states      (B, N, actor_dim)
        • scans             (B, N, scan_dim)
        • actions           (B, N, act_dim)
        • reward            (B, N)
        • not_done          (B, N)
    """

    def __init__(self, critic_dim, actor_dim, action_dim, num_agents,
                 scan_size, max_size=int(1e6)):

        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.num_agents = num_agents
        self.actor_dim = actor_dim
        self.action_dim = action_dim
        self.scan_size = scan_size
        self.critic_dim = critic_dim

        # critic-level data
        self.state = np.zeros((max_size, critic_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, critic_dim), dtype=np.float32)

        # per-agent data
        self.actor_state = np.zeros((max_size, num_agents, actor_dim), dtype=np.float32)
        self.actor_state_next = np.zeros((max_size, num_agents, actor_dim), dtype=np.float32)
        self.scan = np.zeros((max_size, num_agents, scan_size), dtype=np.float32)
        self.scan_next = np.zeros((max_size, num_agents, scan_size), dtype=np.float32)
        self.action = np.zeros((max_size, num_agents, action_dim), dtype=np.float32)

        # reward + termination
        self.reward = np.zeros((max_size, num_agents), dtype=np.float32)
        self.not_done = np.zeros((max_size, num_agents), dtype=np.float32)

    # ------------------------------------------------------
    def add(self,
            state_critic,
            state_actor,
            scan,
            next_scan,
            action,
            next_state_critic,
            next_state_actor,
            reward,
            done):

        i = self.ptr

        # critic features
        self.state[i] = state_critic
        self.next_state[i] = next_state_critic

        # per-agent
        self.actor_state[i] = state_actor
        self.actor_state_next[i] = next_state_actor
        self.scan[i] = scan
        self.scan_next[i] = next_scan
        self.action[i] = action

        # reward + continuation mask
        self.reward[i] = reward
        self.not_done[i] = 1.0 - done

        # pointer update
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # ------------------------------------------------------
    def len(self):
        return self.size

    # ------------------------------------------------------
    def sample(self, batch_size):
        """
        Returns:
            state_critic        (B, critic_dim)
            next_state_critic   (B, critic_dim)
            reward              (B, N)
            not_done            (B, N)
            actor_states        (B, N, actor_dim)
            actor_states_next   (B, N, actor_dim)
            actions             (B, N, action_dim)
            scans               (B, N, scan_dim)
            scans_next          (B, N, scan_dim)
        """

        idx = np.random.randint(0, self.size, size=batch_size)

        # critic
        state_critic = torch.tensor(self.state[idx], dtype=torch.float32)
        next_state_critic = torch.tensor(self.next_state[idx], dtype=torch.float32)

        # per-agent
        actor_states = torch.tensor(self.actor_state[idx], dtype=torch.float32)
        actor_states_next = torch.tensor(self.actor_state_next[idx], dtype=torch.float32)
        actions = torch.tensor(self.action[idx], dtype=torch.float32)

        scans = torch.tensor(self.scan[idx], dtype=torch.float32)
        scans_next = torch.tensor(self.scan_next[idx], dtype=torch.float32)

        reward = torch.tensor(self.reward[idx], dtype=torch.float32)
        not_done = torch.tensor(self.not_done[idx], dtype=torch.float32)

        # device transfer handled in SAC, not here
        return (state_critic,
                next_state_critic,
                reward,
                not_done,
                actor_states,
                actor_states_next,
                actions,
                scans,
                scans_next)


# ==========================================================
# Target Network Wrapper
# ==========================================================
class TargetNet:
    """
    Lightweight wrapper with Polyak averaging.
    """

    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def alpha_sync(self, alpha):
        src = self.model.state_dict()
        tgt = self.target_model.state_dict()

        for k in src:
            tgt[k] = alpha * tgt[k] + (1 - alpha) * src[k]

        self.target_model.load_state_dict(tgt)


# ==========================================================
# Soft Update (SAC)
# ==========================================================
def soft_update(target_net, source_net, tau=1e-3):
    """
    target ← (1 - tau)*target + tau*source
    """
    with torch.no_grad():
        for t, s in zip(target_net.parameters(), source_net.parameters()):
            t.data.copy_((1 - tau) * t.data + tau * s.data)
