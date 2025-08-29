import random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from typing import Dict
from .mouse import MouseGridEnv, ACTIONS, WALL

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ---------------- models ----------------
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*5*5, 64), nn.ReLU(),
            nn.Linear(64, 4)   # 4 actions
        )
    def forward(self, x): return self.net(x)

class RewardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*5*5, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

# ---------------- helpers ----------------
def returns(rews: np.ndarray, gamma=0.99):
    """discounted returns. tiny util, but easy to mess up if copied wrong."""
    G = 0.0; out = []
    for r in reversed(rews):
        G = float(r) + gamma * G
        out.append(G)
    return np.array(list(reversed(out)), np.float32)

def kl_categorical(p_logits, q_logits):
    """KL( policy || reference ). we use this as a 'dont drift too far' penalty."""
    p_logp = torch.log_softmax(p_logits, -1)
    q_logp = torch.log_softmax(q_logits, -1)
    p = p_logp.exp()
    return (p * (p_logp - q_logp)).sum(-1)

def has_movement(frames_list):
    if not frames_list or len(frames_list) < 2:
        return False
    first = np.array(frames_list[0])
    for f in frames_list[1:]:
        if not np.array_equal(first, np.array(f)):
            return True
    return False

# ---------------- rollout (training) ----------------
@torch.no_grad()
def rollout(env: MouseGridEnv, policy: PolicyNet, max_steps=60, device="cpu") -> Dict[str, np.ndarray]:
    # we include an initial no-op frame so the UI always shows something
    obs_list, acts, rews, grids, kinds = [], [], [], [], []
    obs = env.reset()
    grids.append(env.grid.copy()); kinds.append(0); obs_list.append(obs); rews.append(0.0); acts.append(-1)
    for _ in range(max_steps):
        t_obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits = policy(t_obs)
        a = int(torch.distributions.Categorical(logits=logits).sample().item())
        step = env.step(a)
        obs_list.append(obs); acts.append(a); rews.append(step.reward)
        grids.append(env.grid.copy()); kinds.append(step.info["prev"])
        obs = step.obs
        if step.done: break
    return {"obs": np.array(obs_list, np.float32),
            "acts": np.array(acts, np.int64),
            "rews": np.array(rews, np.float32),
            "grids": np.array(grids, np.int8),
            "kinds": np.array(kinds, np.int64)}

# ---------------- rollout (UI-friendly) ----------------
@torch.no_grad()
def rollout_ui(env: MouseGridEnv, policy: PolicyNet, max_steps=60, device="cpu"):
    """
    For visualization only: try a few alt actions if the first one is a wall/bounds bonk.
    Keeps the movie moving so humans can actualy see the behavior.
    """
    obs_list, acts, rews, grids, kinds = [], [], [], [], []
    obs = env.reset()
    grids.append(env.grid.copy()); kinds.append(0); obs_list.append(obs); rews.append(0.0); acts.append(-1)

    for _ in range(max_steps):
        # propose from policy
        t_obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits = policy(t_obs)
        a = int(torch.distributions.Categorical(logits=logits).sample().item())
        # nudge away from walls if possible
        i, j = env.mouse_pos
        tried = set()
        for _tries in range(8):
            if a in tried:
                a = int(np.random.randint(0, 4))
            tried.add(a)
            di, dj = ACTIONS[a]; ni, nj = i + di, j + dj
            if 0 <= ni < env.size and 0 <= nj < env.size and env.grid[ni, nj] != WALL:
                break
        step = env.step(a)
        obs_list.append(obs); acts.append(a); rews.append(step.reward)
        grids.append(env.grid.copy()); kinds.append(step.info["prev"])
        obs = step.obs
        if step.done: break

    return {"obs": np.array(obs_list, np.float32),
            "acts": np.array(acts, np.int64),
            "rews": np.array(rews, np.float32),
            "grids": np.array(grids, np.int8),
            "kinds": np.array(kinds, np.int64)}

# ---------------- Task 1: REINFORCE baseline ----------------
def train_reinforce(episodes=100, lr=1e-3, gamma=0.99, device="cpu", seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = MouseGridEnv(); policy = PolicyNet().to(device); opt = optim.Adam(policy.parameters(), lr=lr)
    ep_returns = []
    for _ in range(episodes):
        traj = rollout(env, policy, device=device)

        # mask off the initial no-op (-1)
        mask = traj["acts"] >= 0
        obs_np  = traj["obs"][mask]
        acts_np = traj["acts"][mask]
        rews_np = traj["rews"][mask]

        G = returns(rews_np, gamma)
        obs  = torch.tensor(obs_np,  dtype=torch.float32, device=device)
        acts = torch.tensor(acts_np, dtype=torch.int64,   device=device)
        Gt   = torch.tensor(G,       dtype=torch.float32, device=device)

        logits = policy(obs)
        logp   = torch.log_softmax(logits, -1)
        act_lp = logp.gather(1, acts.view(-1,1)).squeeze(1)
        adv = Gt - Gt.mean()
        loss = -(adv * act_lp).mean()

        opt.zero_grad(); loss.backward(); opt.step()
        ep_returns.append(float(rews_np.sum()))
    torch.save(policy.state_dict(), ARTIFACTS / "policy_reinforce.pt")
    return policy, ep_returns

# ---------------- Task 2: prefs -> Bradley–Terry reward ----------------
def make_pref_pairs_no_org(trajs, seg_len=6, max_pairs=500):
    """
    Fake user: prefer segments with FEWER organic cheese hits.
    """
    ORG = 5; pairs = []; rng = random.Random(0)
    for _ in range(min(max_pairs, len(trajs)*5)):
        t1, t2 = rng.choice(trajs), rng.choice(trajs)
        if len(t1["rews"]) < 2 or len(t2["rews"]) < 2: continue
        i1 = rng.randrange(0, max(1, len(t1["rews"])-seg_len+1))
        i2 = rng.randrange(0, max(1, len(t2["rews"])-seg_len+1))
        o1, k1 = t1["obs"][i1:i1+seg_len], t1["kinds"][i1:i1+seg_len]
        o2, k2 = t2["obs"][i2:i2+seg_len], t2["kinds"][i2:i2+seg_len]
        n1 = int(np.sum(k1 == 5)); n2 = int(np.sum(k2 == 5))
        if n1 == n2: continue
        y = 1 if n1 < n2 else 0
        pairs.append((o1, o2, y))
    return pairs

def fit_bradley_terry(pairs, epochs=3, lr=1e-3, device="cpu"):
    reward = RewardNet().to(device)
    opt = optim.Adam(reward.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        random.shuffle(pairs)
        for o1, o2, y in pairs:
            o1 = torch.tensor(o1, dtype=torch.float32, device=device)
            o2 = torch.tensor(o2, dtype=torch.float32, device=device)
            y  = torch.tensor([y], dtype=torch.float32, device=device)
            r1 = reward(o1).sum(); r2 = reward(o2).sum()
            logit = r1 - r2
            loss = loss_fn(logit.view(1), y)
            opt.zero_grad(); loss.backward(); opt.step()
    torch.save(reward.state_dict(), ARTIFACTS / "reward_model.pt")
    return reward

# ---------------- Task 3: RLHF (learned reward + KL) ----------------
def train_rlhf(policy_ref: PolicyNet, reward_model: RewardNet,
               steps=80, beta=0.02, lr=1e-3, gamma=0.99, device="cpu"):
    env = MouseGridEnv()
    policy = PolicyNet().to(device)
    policy.load_state_dict(policy_ref.state_dict())
    opt = optim.Adam(policy.parameters(), lr=lr)

    for _ in range(steps):
        traj = rollout(env, policy, device=device)

        mask = traj["acts"] >= 0
        obs_np  = traj["obs"][mask]
        acts_np = traj["acts"][mask]

        obs  = torch.tensor(obs_np,  dtype=torch.float32, device=device)
        acts = torch.tensor(acts_np, dtype=torch.int64,   device=device)

        with torch.no_grad():
            r_hat = reward_model(obs)
            ref_logits = policy_ref(obs)

        logits = policy(obs)
        kl = kl_categorical(logits, ref_logits).detach()
        shaped = r_hat - beta * kl  # human reward minus drift penalty

        G  = returns(shaped.cpu().numpy(), gamma)
        Gt = torch.tensor(G, dtype=torch.float32, device=device)

        logp = torch.log_softmax(logits, -1).gather(1, acts.view(-1,1)).squeeze(1)
        adv  = Gt - Gt.mean()
        loss = -(adv * logp).mean()

        opt.zero_grad(); loss.backward(); opt.step()

    torch.save(policy.state_dict(), ARTIFACTS / "policy_rlhf.pt")
    return policy

# ---------------- loaders ----------------
def load_policy(kind: str, device="cpu") -> PolicyNet:
    policy = PolicyNet().to(device)
    path = ARTIFACTS / ("policy_reinforce.pt" if kind == "base" else "policy_rlhf.pt")
    if path.exists():
        policy.load_state_dict(torch.load(path, map_location="cpu"))
    policy.eval()
    return policy
